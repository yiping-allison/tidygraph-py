supported_packaging_systems := "uv nix"

has_nix := `command -v nix`

[private]
[default]
list:
    @just --list --unsorted

[private]
ensure_nix:
    #!/usr/bin/env sh
    if [ -z "{{ has_nix }}" ]; then
        echo "❌ Nix must be installed";
        exit 1;
    fi;

[private]
build_nix: ensure_nix
    @echo "📦️ Packaging using nix..."
    @nix build --no-link

[private]
build_uv:
    @echo "📦️ Packaging using uv..."
    @uv build

# Build project
[group('build')]
build *system='uv':
    #!/usr/bin/env sh
    if echo "{{ supported_packaging_systems }}" | grep -qw "{{ system }}"; then
        system_target="build_{{ system }}";
        just "$system_target";
    else
        echo "❌ Unsupported packaging system: {{ system }}";
        exit 1;
    fi;

# Run formatters
[group('misc')]
fmt:
    @uv run ruff format src tests;
    @uv run ruff check src tests --fix;
    @if [ -n "{{ has_nix }}" ]; then nix fmt flake.nix nix; fi

# Display flake schema
[group('misc')]
schema: ensure_nix
    @nix flake show

# Run tests
[group('misc')]
test:
    @uv run --frozen --no-default-groups --group test pytest tests
