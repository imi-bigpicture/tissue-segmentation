	{
	description = "Developer shell for Tissue Segmentation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-25.05";
  };

	outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { 
        inherit system;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          uv
        ];
      };
    };
}
