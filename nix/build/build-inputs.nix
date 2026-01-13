{
  pkgs,
  lib,
  stdenv,
}:
[
]
++ lib.optionals stdenv.hostPlatform.isLinux [
  stdenv.cc.libc.dev
]
