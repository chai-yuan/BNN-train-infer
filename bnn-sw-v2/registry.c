/* registry.c */
#include <string.h>
#include "bnne-internal.h"

#define BNNE_ARCH(id) extern const bnne_arch bnne_arch_##id;
#include "bnne-archs.def"
#undef BNNE_ARCH

static const bnne_arch* const k_archs[] = {
    #define BNNE_ARCH(id) &bnne_arch_##id,
    #include "bnne-archs.def"
    #undef BNNE_ARCH
};

const bnne_arch* bnne_arch_find(const char* name) {
    if (!name)
        return NULL;
    for (size_t i = 0; i < sizeof k_archs / sizeof k_archs[0]; ++i)
        if (strcmp(k_archs[i]->name, name) == 0)
            return k_archs[i];
    return NULL;
}
