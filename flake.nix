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
          overrides = final: prev: {};

          pythonSet = (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (
            pkgs.lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              workspaceOverlay
              overrides
            ]
          );

          editable = pythonSet.overrideScope editableOverlay;

          # metadata
          # ! NOTE: projectName must match the name in pyproject.toml
          projectName = "tidygraph";
          nixPackage = pythonSet.${projectName};
        in
          f {
            inherit pkgs workspace pythonSet editable projectName nixPackage;
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
        ...
      }: let
        inherit (pkgs.callPackage pyproject-nix.build.util {}) mkApplication;
        venv = pythonSet.mkVirtualEnv "${projectName}" workspace.deps.default;
      in {
        default = mkApplication {
          venv = venv;
          package = nixPackage;
        };
      }
    );
  };
}
