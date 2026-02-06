supported_packaging_tools := "uv nix"

has_nix := `command -v nix`

[default]
[private]
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
    @uv build --clear

# Build package using tool
[group('build')]
build *tool='uv':
    #!/usr/bin/env sh
    if echo "{{ supported_packaging_tools }}" | grep -qw "{{ tool }}"; then
        tool_target="build_{{ tool }}";
        just "$tool_target";
    else
        echo "❌ Unsupported packaging tool: {{ tool }}";
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

# Publish package to `test.pypi` index for testing
[group('misc')]
upload: build
    @uv publish --index testpypi
