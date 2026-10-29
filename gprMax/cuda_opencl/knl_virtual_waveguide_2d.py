"""Live-plane device kernels for reduced TE/TM virtual-waveguide apertures.

The auxiliary grid retains the parent's Cartesian axes. Invariant storage
planes are not physical walls. Each thread owns one transverse Yee sample;
rear clearing is a separate launch, after the aperture update has read H.
"""

from string import Template

from gprMax.cuda_opencl import knl_virtual_waveguide as full


def specification(original, geometry):
    """Specialise the shared device ABI to a reduced Yee live plane."""
    electric = original is full.couple_electric or original is full.clear_rear_electric
    clear = original is full.clear_rear_electric or original is full.clear_rear_magnetic
    names = geometry.active_electric if electric else geometry.active_magnetic
    invariant, live = geometry.invariant_axis, geometry.live_index
    body = ["$CUDA_IDX", "if (i < NPOINTS) {"]
    for normal in range(3):
        if normal == invariant:
            continue
        transverse = 3 - normal - invariant
        axes = [a for a in range(3) if a != normal]
        lower, upper = ("u0", "u1") if axes[0] == transverse else ("v0", "v1")
        body.append(f"if (normal_axis == {normal}) {{")
        body.append(f"int count = {upper} - {lower};")
        if clear:
            body.append("int face = (u1-u0+1)*(v1-v0+1); int offset = i / face; int q = i % face;")
            body.append("int n = (direction_sign < 0 ? plane_index : plane_index-1) + offset;")
        else:
            body.append("int q = i;")
        sizes = ["aux_nx", "aux_ny", "aux_nz"]
        body.append(f"int aperture = direction_sign < 0 ? 0 : {sizes[normal]};")
        body.append("int inside = direction_sign < 0 ? 0 : aperture-1;")

        def idx(n, q="q", main=False):
            coords = [str(live)] * 3
            coords[normal] = n
            coords[transverse] = f"({lower}+({q}))" if main else q
            x, y, z = coords
            return f"IDX3D_FIELDS({x},{y},{z})" if main else f"(({x})*(aux_ny+1)*(aux_nz+1)+({y})*(aux_nz+1)+({z}))"

        for name in names:
            axis = "xyz".index(name[1].lower())
            stop = int(axis != transverse) if electric else int(axis == transverse)
            body.append(f"if (q < count+{stop}) {{")
            if clear:
                normal_stop = ["$NX_FIELDS", "$NY_FIELDS", "$NZ_FIELDS"][normal]
                excluded = int(axis == normal) if electric else int(axis != normal)
                start = "plane_index+1" if electric else f"plane_index+{int(axis == normal)}"
                end = f"plane_index-{int(electric and axis == normal)}"
                body.append(
                    f"if ((direction_sign > 0 && n < {end}) || (direction_sign < 0 && n >= {start} && n < {normal_stop}-{excluded})) main_{name}[{idx('n', main=True)}] = 0;"
                )
            elif not electric:
                if axis == normal:
                    body.append(f"aux_{name}[{idx('aperture')}] = main_{name}[{idx('plane_index', main=True)}];")
            elif axis == normal:
                body.append(
                    f"main_{name}[{idx('(direction_sign < 0 ? plane_index : plane_index-1)', main=True)}] = aux_{name}[{idx('inside')}];"
                )
            else:
                h = "H" + "xyz"[3 - axis - normal]
                sign = 1 if (normal - axis) % 3 == 1 else -1
                select = "q > 0 && q < count" if axis == invariant else "q < count"
                body.append(f"int a = {idx('aperture')}; if ({select}) {{")
                body.append(f"unsigned int material = aux_ID[{axis}*(aux_nx+1)*(aux_ny+1)*(aux_nz+1)+a];")
                body.append(
                    f"$REAL difference = direction_sign < 0 ? aux_{h}[{idx('0')}] - main_{h}[{idx('plane_index-1', main=True)}] : main_{h}[{idx('plane_index', main=True)}] - aux_{h}[{idx('inside')}];"
                )
                body.append(
                    f"$REAL value = aux_coeffs[IDX2D_MAT(material,0)]*aux_{name}[a] + ({sign})*aux_coeffs[IDX2D_MAT(material,{normal+1})]*difference;"
                )
                if axis == invariant:
                    hn = "H" + "xyz"[normal]
                    body.append(
                        f"value -= ({sign})*aux_coeffs[IDX2D_MAT(material,{transverse+1})]*(aux_{hn}[a]-aux_{hn}[{idx('aperture', 'q-1')}]);"
                    )
                body.append(f"aux_{name}[a] = value; }} main_{name}[{idx('plane_index', main=True)}] = aux_{name}[a];")
            body.append("}")
        body.append("}")
    body.append("}")
    return {**original, "func": Template("\n".join(body))}
