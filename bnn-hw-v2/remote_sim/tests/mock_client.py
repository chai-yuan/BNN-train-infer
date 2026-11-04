#!/usr/bin/env python3
"""Minimal TCP client for bnn-hw-v2/remote_sim.

Plays the role of the upper software stack: sends the committed Qwen3 command
stream one instruction at a time, serves the simulator's memory reads/writes,
and writes the final memory image for comparison with simple_test.

Wire format (little-endian, 16-byte header + payload):
    magic  u32 = 0x424E4E53 ("BNNS")
    type   u8
    flags  u8
    seq    u16
    addr   u32
    length u32

READ requests always fetch 8 bytes and carry no payload; READ responses carry
8 bytes. READ_BURST requests carry a 4-byte byte count and are answered with
the whole data payload. WRITE messages carry 8 bytes with the byte strobe in
flags.
"""

import argparse
import os
import socket
import struct
import subprocess
import sys
import time

MAGIC = 0x424E4E53
HEADER = struct.Struct("<IBBHII")

TYPE_CMD = 0x01
TYPE_READ = 0x02
TYPE_WRITE = 0x03
TYPE_QUIT = 0x04
TYPE_READ_BURST = 0x05
TYPE_READ_RSP = 0x81
TYPE_CMD_DONE = 0x82
TYPE_QUIT_ACK = 0x83
TYPE_READ_BURST_RSP = 0x85


class ProtocolError(RuntimeError):
    pass


def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("connection closed by simulator")
        data += chunk
    return bytes(data)


def recv_message(sock):
    header = recv_exact(sock, HEADER.size)
    magic, mtype, flags, seq, addr, length = HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:08x}")
    return {
        "type": mtype,
        "flags": flags,
        "seq": seq,
        "addr": addr,
        "length": length,
        "payload": recv_exact(sock, length) if length else b"",
    }


def send_message(sock, mtype, seq=0, addr=0, flags=0, payload=b""):
    sock.sendall(HEADER.pack(MAGIC, mtype, flags, seq, addr, len(payload)) + payload)


def parse_commands(path):
    commands = []
    with open(path, "r", encoding="ascii") as stream:
        for line in stream:
            token = line.split("#", 1)[0].strip()
            if not token:
                continue
            if token[:2].lower() == "0x":
                token = token[2:]
            commands.append(int(token, 16))
    return commands


def load_memory(path):
    with open(path, "rb") as stream:
        image = stream.read()
    memory = bytearray(max(1 << 20, (len(image) + 7) & ~7))
    memory[: len(image)] = image
    return memory, len(image)


def choose_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def start_server(server, port):
    process = subprocess.Popen(
        [server, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        if "listening on" in line:
            return process
        if not line.strip():
            continue
        print(f"server: {line.rstrip()}")
    output = process.stdout.read()
    process.wait(timeout=5)
    raise RuntimeError(
        f"simulator exited before listening (code {process.returncode}): {output}"
    )


def connect(port, timeout=10.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            sock.settimeout(60.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def apply_write(memory, message, highest):
    address = message["addr"]
    payload = message["payload"]
    if address + len(payload) > len(memory):
        raise ProtocolError(f"write beyond memory at 0x{address:x}")
    for index, value in enumerate(payload):
        if message["flags"] & (1 << index):
            memory[address + index] = value
    return max(highest, address + len(payload))


def serve_message(sock, memory, message, highest):
    if message["type"] == TYPE_READ:
        address = message["addr"]
        # READ requests always fetch 8 bytes; the header carries no payload.
        length = 8
        if address + length > len(memory):
            raise ProtocolError(f"read beyond memory at 0x{address:x}")
        send_message(
            sock,
            TYPE_READ_RSP,
            seq=message["seq"],
            addr=address,
            payload=bytes(memory[address : address + length]),
        )
        return highest
    if message["type"] == TYPE_READ_BURST:
        if len(message["payload"]) != 4:
            raise ProtocolError("burst read request without a byte count")
        address = message["addr"]
        length = struct.unpack("<I", message["payload"])[0]
        if length == 0 or address + length > len(memory):
            raise ProtocolError(f"burst read beyond memory at 0x{address:x}")
        send_message(
            sock,
            TYPE_READ_BURST_RSP,
            seq=message["seq"],
            addr=address,
            payload=bytes(memory[address : address + length]),
        )
        return highest
    if message["type"] == TYPE_WRITE:
        return apply_write(memory, message, highest)
    raise ProtocolError(f"unexpected message type 0x{message['type']:02x}")


def run(sock, commands, memory):
    """Send commands one at a time, serving simulator memory traffic."""
    highest = 0
    for index, inst in enumerate(commands):
        send_message(
            sock, TYPE_CMD, seq=index & 0xFFFF, payload=struct.pack("<Q", inst)
        )
        while True:
            message = recv_message(sock)
            if message["type"] == TYPE_CMD_DONE:
                if message["seq"] != (index & 0xFFFF):
                    raise ProtocolError(
                        f"command {index}: completion seq {message['seq']}"
                    )
                if message["flags"] & 1:
                    raise ProtocolError(f"command {index}: simulator reported cmd_error")
                break
            highest = serve_message(sock, memory, message, highest)

    send_message(sock, TYPE_QUIT)
    while True:
        message = recv_message(sock)
        if message["type"] == TYPE_QUIT_ACK:
            break
        highest = serve_message(sock, memory, message, highest)
    return highest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="path to the remote_sim binary")
    parser.add_argument("--port", type=int, default=0, help="TCP port (0 = pick free)")
    parser.add_argument("--commands", required=True, help="commands.hex")
    parser.add_argument("--memory", required=True, help="initial memory image")
    parser.add_argument("--output", required=True, help="output memory image")
    parser.add_argument("--reference", help="optional reference memory image")
    args = parser.parse_args()

    commands = parse_commands(args.commands)
    memory, initial_size = load_memory(args.memory)
    port = args.port or choose_port()

    server = start_server(args.server, port)
    try:
        sock = connect(port)
        with sock:
            highest = run(sock, commands, memory)
        server.wait(timeout=30)
        output = server.stdout.read()
        if output.strip():
            print(output.rstrip())
        if server.returncode != 0:
            raise RuntimeError(f"simulator exited with code {server.returncode}")
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()

    output_size = min(len(memory), max(initial_size, highest))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as stream:
        stream.write(memory[:output_size])
    print(f"commands: {len(commands)}, wrote {output_size} bytes -> {args.output}")

    if args.reference:
        with open(args.reference, "rb") as stream:
            reference = stream.read()
        if reference != memory[:output_size]:
            limit = min(len(reference), output_size)
            mismatch = next(
                (i for i in range(limit) if reference[i] != memory[i]), None
            )
            detail = "size mismatch" if mismatch is None else f"first mismatch at {mismatch}"
            raise SystemExit(f"FAIL: {detail} (reference {len(reference)} bytes)")
        print(f"PASS: output matches {args.reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
