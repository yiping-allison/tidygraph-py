[private]
default:
    @just --list --unsorted

supported_packaging_systems := "uv nix"

_build_nix:
    @echo "📦️ Packaging using nix..."
    @nix build --no-link

_build_uv:
    @echo "📦️ Packaging using uv..."
    @uv build

# Build project
[group('build')]
build *system='uv':
    #!/usr/bin/env sh
    if echo "{{ supported_packaging_systems }}" | grep -qw "{{ system }}"; then
        system_target="_build_{{system}}";
        just "$system_target";
    else
        echo "❌ Unsupported packaging system: {{ system }}";
        exit 1;
    fi;

# Run formatters
[group('misc')]
fmt:
    @uv run ruff format src tests
    @uv run ruff check src tests --fix
    @nix fmt flake.nix nix

# Display flake schema
[group('misc')]
schema:
    @nix flake show

# Run tests
[group('misc')]
test:
    @uv run pytest tests
