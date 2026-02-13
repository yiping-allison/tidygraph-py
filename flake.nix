{
  description = "tidy-like interface for network manipulation and visualization in Python";

  inputs = {
    flake-schemas.url = "https://flakehub.com/f/DeterminateSystems/flake-schemas/*";

    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/*";

    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
  };

  outputs = {
    self,
    flake-schemas,
    nixpkgs,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-darwin" "x86_64-darwin" "aarch64-linux"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (
        system: let
          pkgs = import nixpkgs {
            inherit system;
          };

          # ! NOTE: projectName and version must match values in pyproject.toml
          config = pkgs.lib.importTOML ./pyproject.toml;
          projectName = config.project.name;
          version = config.project.version;

          python = pkgs.python313;
          pythonPackages = python.pkgs;

          workspace = uv2nix.lib.workspace.loadWorkspace {
            workspaceRoot = ./.;
          };

          workspaceOverlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          editableOverlay = workspace.mkEditablePyprojectOverlay {
            # ! NOTE: This requires that the REPO_ROOT environment variable is set before the flake is run.
            # ! This is automatically done if you use direnv with the provided .envrc file.
            root = "$REPO_ROOT";
          };

          # Use pre-built packages from nixpkgs
          hacks = pkgs.callPackage pyproject-nix.build.hacks {};

          # Add additional build overrides
          overrides = final: prev: {
            cairocffi = hacks.nixpkgsPrebuilt {
              from = pythonPackages.cairocffi;
              prev = prev.cairocffi;
            };

            # ! NOTE: This does not match the same version required by pyproject.toml
            # See https://github.com/NixOS/nixpkgs/blob/nixos-25.11/pkgs/development/python-modules/kaleido/default.nix#L96.
            kaleido = hacks.nixpkgsPrebuilt {
              from = pythonPackages.kaleido;
              prev = prev.kaleido;
            };
          };

          pythonSet = (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (
            pkgs.lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              workspaceOverlay
              overrides
            ]
          );

          sourceOverride = final: prev: {
            "${projectName}" = prev.${projectName}.overrideAttrs (old: {
              src = pkgs.lib.fileset.toSource {
                root = old.src;
                fileset = pkgs.lib.fileset.unions [
                  ./pyproject.toml
                  ./README.md
                  ./src/${projectName}/__init__.py
                ];
              };
            });
          };

          editable = pythonSet.overrideScope (
            pkgs.lib.composeManyExtensions [
              editableOverlay
              sourceOverride
            ]
          );

          nixPackage = pythonSet.${projectName};
        in
          f {
            inherit pkgs workspace pythonSet editable projectName version nixPackage;
          }
      );
  in {
    schemas = flake-schemas.schemas;

    formatter = forEachSupportedSystem ({pkgs, ...}: pkgs.alejandra);

    devShells = forEachSupportedSystem (
      {
        pkgs,
        workspace,
        editable,
        pythonSet,
        projectName,
        ...
      }: let
        venv = editable.mkVirtualEnv "${projectName}-dev" workspace.deps.all;
      in {
        default = pkgs.callPackage ./nix/devShell.nix {
          inherit pythonSet venv;
        };
      }
    );

    packages = forEachSupportedSystem (
      {
        pkgs,
        workspace,
        pythonSet,
        nixPackage,
        projectName,
        version,
        ...
      }: let
        inherit (pkgs.callPackage pyproject-nix.build.util {}) mkApplication;
        venv = pythonSet.mkVirtualEnv "${projectName}-${version}" workspace.deps.default;
      in {
        default = mkApplication {
          venv = venv;
          package = nixPackage;
        };
      }
    );
  };
}
