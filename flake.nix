{
  description = "Helios Simulator Dev Flake";

  # Flake inputs
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs"; # also valid: "nixpkgs"
  };

  # Flake outputs
  outputs = { self, nixpkgs }:
    let
      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
      ];

      # Helper to provide system-specific attributes
      forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      });
    in
    {
      # Development environment output
      devShells = forAllSystems ({ pkgs }: {
        default = pkgs.mkShell {
          # The Nix packages provided in the environment
          packages = with pkgs; [
            bash
            cmake
            gnumake
            gcc10
            
            mpich
            mold
            
            # cudatoolkit linuxPackages.nvidia_x11
            # cudatoolkit linuxPackages.nvidiaPackages
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.cudnn
            cudaPackages.cuda_nsight           
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.cudatoolkit.lib}/lib:${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH"
            export CUDA_HOME="${pkgs.cudatoolkit}"
            export CUDA_PATH="${pkgs.cudatoolkit}"
            export NVCC_GENCODE="-gencode=arch=compute_61,code=sm_61"
            export MPI_HOME="${pkgs.mpich}"
            export GCC_HOME="${pkgs.gcc10}"
            export MOLD_HOME="${pkgs.mold}"
          '';
        };
      });
    };
}
