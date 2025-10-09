"""Debug interface charge implementation"""

import json
import os


def check_all_interface_tags(out_dir):
    """Check all interface facet tags"""

    from main import GeometricParameters, create_mesh

    with open(os.path.join(out_dir, "parameters.json")) as f:
        params = json.load(f)

    geom_params = GeometricParameters(**params["geometric"])
    msh, cell_tags, facet_tags = create_mesh(geom_params)

    # Check which cells are on each side
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    # Get all unique facet tags
    unique_tags = set(facet_tags.values)
    print(f"All facet tags: {sorted(unique_tags)}")

    # Check each tag
    for tag in sorted(unique_tags):
        interface_facets = facet_tags.find(tag)
        if len(interface_facets) == 0:
            continue

        print(f"\n=== Facet Tag {tag} ({len(interface_facets)} facets) ===")

        # Sample first facet
        facet = interface_facets[0]
        cells = f_to_c.links(facet)

        if len(cells) == 2:  # Internal facet
            tags = [cell_tags.values[c] for c in cells]
            materials = []
            for t in tags:
                if t == 1:
                    materials.append("SiC")
                elif t == 2:
                    materials.append("SiO2")
                elif t == 3:
                    materials.append("Vac")
                else:
                    materials.append(f"Unknown({t})")
            print(f"  Internal interface: {materials[0]} <-> {materials[1]}")
        else:  # Boundary facet
            cell_tag = cell_tags.values[cells[0]]
            material = "SiC" if cell_tag == 1 else "SiO2" if cell_tag == 2 else "Vac"
            print(f"  External boundary: {material}")


if __name__ == "__main__":
    check_all_interface_tags("example_output")
