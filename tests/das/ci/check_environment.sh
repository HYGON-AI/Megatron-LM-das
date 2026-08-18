#!/usr/bin/env bash
# Environment self-check (runs inside the test container):
#   - DTK env (DTK_ROOT overridable, default /opt/dtk)
#   - torch importable, >= 8 GPUs visible
#   - transformer_engine presence (missing -> warning; nightly training needs it)
#   - coverage presence (missing -> sets DAS_COVERAGE_DISABLED=1, plain pytest)
set -euo pipefail

DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [[ -d "${DTK_ROOT}" ]]; then
    echo "DTK root: ${DTK_ROOT}"
    if [[ -f "${DTK_ROOT}/env.sh" ]]; then
        # shellcheck disable=SC1091
        source "${DTK_ROOT}/env.sh"
    fi
    export PATH="${DTK_ROOT}/bin:${PATH}"
    export LD_LIBRARY_PATH="${DTK_ROOT}/lib:${DTK_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
else
    echo "WARN: DTK root not found at ${DTK_ROOT} (override with DTK_ROOT if needed)"
fi

python3 - <<'PY'
import sys

import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
count = torch.cuda.device_count()
print(f"gpus={count}")
assert count >= 8, f"expected >= 8 GPUs, got {count}"
try:
    import transformer_engine

    print(f"transformer_engine={transformer_engine.__version__}")
except ImportError:
    print(
        "transformer_engine: NOT INSTALLED "
        "(unit tests may use local paths; the default functional smoke requires it)"
    )
PY

if ! python3 -c "import coverage" >/dev/null 2>&1; then
    echo "coverage: NOT INSTALLED -> DAS_COVERAGE_DISABLED=1"
    echo "DAS_COVERAGE_DISABLED=1" >> "${GITHUB_ENV:-/dev/null}"
fi

echo "Environment check passed."
