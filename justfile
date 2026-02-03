[private]
default:
    @just --list --unsorted

supported_packaging_systems := "uv nix"
has_nix := `command -v nix`

_build_nix:
    #!/usr/bin/env sh
    echo "📦️ Packaging using nix..."
    if [ -z "{{ has_nix }}" ]; then
        echo "❌ Nix must be installed";
        exit 1;
    fi;
    nix build --no-link;

_build_uv:
    @echo "📦️ Packaging using uv..."
    @uv build

# Build project
[group('build')]
build *system='uv':
    #!/usr/bin/env sh
    if echo "{{ supported_packaging_systems }}" | grep -qw "{{ system }}"; then
        system_target="_build_{{ system }}";
        just "$system_target";
    else
        echo "❌ Unsupported packaging system: {{ system }}";
        exit 1;
    fi;

# Run formatters
[group('misc')]
fmt:
    #!/usr/bin/env sh
    uv run ruff format src tests;
    uv run ruff check src tests --fix;

    if [ -n "{{ has_nix }}" ]; then
        nix fmt flake.nix nix
    fi;

# Display flake schema
[group('misc')]
schema:
    #!/usr/bin/env sh
    if [ -z "{{ has_nix }}" ]; then
        echo "❌ Nix must be installed";
        exit 1;
    fi;
    nix flake show;

# Run tests
[group('misc')]
test:
    @uv run --frozen --no-default-groups pytest tests
