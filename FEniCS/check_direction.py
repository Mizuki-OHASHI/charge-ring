"""Debug interface charge implementation"""

import json
import os


def check_interface_normal_direction(out_dir):
    """Check which side is '+' and '-' at the SiC/SiO2 interface"""

    # Load mesh
    from main import GeometricParameters, create_mesh

    with open(os.path.join(out_dir, "parameters.json")) as f:
        params = json.load(f)

    geom_params = GeometricParameters(**params["geometric"])
    msh, cell_tags, facet_tags = create_mesh(geom_params)

    # Find cells adjacent to interface (tag 15)
    interface_facets = facet_tags.find(15)

    # Check which cells are on each side
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    for facet in interface_facets[:5]:  # Check first few facets
        cells = f_to_c.links(facet)
        print(f"Facet {facet} connects cells: {cells}")
        for cell in cells:
            cell_tag = cell_tags.values[cell]
            material = "SiC" if cell_tag == 1 else "SiO2" if cell_tag == 2 else "Vac"
            print(f"  Cell {cell}: tag={cell_tag} ({material})")


if __name__ == "__main__":
    check_interface_normal_direction("example_output")
