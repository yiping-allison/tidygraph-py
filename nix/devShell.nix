{
  mkShell,
  lib,
  stdenv,
  pkgs,
  pythonSet,
  venv,
}: let
  ld_library_path = import ./build/ld-library-path.nix {inherit pkgs lib stdenv;};
in
  mkShell {
    LD_LIBRARY_PATH = ld_library_path;

    env = {
      UV_NO_SYNC = "1";
      UV_PYTHON = pythonSet.python.interpreter;
      UV_PYTHON_DOWNLOADS = "never";
    };

    packages = [
      # Virtual environment
      venv

      # Development utilities
      pkgs.just
      pkgs.uv
    ];

    shellHook = ''
      unset PYTHONPATH
    '';
  }
