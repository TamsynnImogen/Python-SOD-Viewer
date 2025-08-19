"""
SOD Viewer (pyglet 2.x)
- Open with **O**, set base with **L**
- Texture roots (case-insensitive): <base>/sod, <base>/textures, <base>/textures/rgb
- Formats: .tga / .dds (case-insensitive)
- **B** toggle Borg branch (node 'borg' or containing 'borg' + descendants)
- **A** play/pause transform animations (linear interp @ ~30 FPS)
- **S** Save As: choose Armada 1 (v1.8) or Armada 2 (v1.93) SOD

Controls:
  O open .SOD | L pick base | S save as SOD (1.8/1.93) | B toggle Borg | A play/pause
  L-drag orbit | Scroll zoom | W wire | T textures | C cull | F flip front | D dump | Space reset | Esc quit
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys, struct, math, ctypes, argparse
from collections import defaultdict
from pyglet import shapes  # add near your other pyglet imports inside run_viewer
from pyglet.graphics.shader import Shader, ShaderProgram
from SOD import SOD, Node

# ---------- Math helpers ----------

def mat34_to_blender(mat34):
    m = [[0,0,0,0] for _ in range(4)]
    m[0][0], m[1][0], m[2][0], m[3][0] = mat34[0], mat34[1], mat34[2], 0.0
    m[0][1], m[1][1], m[2][1], m[3][1] = mat34[3], mat34[4], mat34[5], 0.0
    m[0][2], m[1][2], m[2][2], m[3][2] = mat34[6], mat34[7], mat34[8], 0.0
    m[0][3], m[1][3], m[2][3], m[3][3] = mat34[9], mat34[10], mat34[11], 1.0
    for r in range(4): m[0][r] *= -1.0
    for r in range(4): m[r][0] *= -1.0
    return m

def mat_mul(a, b):
    out = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j] + a[i][3]*b[3][j]
    return out

def transform_point(m, v):
    x = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2] + m[0][3]
    y = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2] + m[1][3]
    z = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2] + m[2][3]
    return (x, y, z)

def compute_bounds(pts):
    if not pts: return (0,0,0),(1,1,1)
    xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    return (min(xs),min(ys),min(zs)), (max(xs),max(ys),max(zs))

def mat_perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = [[0]*4 for _ in range(4)]
    m[0][0] = f / aspect
    m[1][1] = f
    m[2][2] = (zfar + znear) / (znear - zfar)
    m[2][3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3][2] = -1.0
    return m

def vec_sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vec_dot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def vec_cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def vec_norm(v):
    l = math.sqrt(max(1e-12, vec_dot(v,v))); return (v[0]/l, v[1]/l, v[2]/l)

def mat_look_at(eye, center, up):
    f = vec_norm(vec_sub(center, eye))
    s = vec_norm(vec_cross(f, up))
    u = vec_cross(s, f)
    m = [[0]*4 for _ in range(4)]
    m[0][0], m[0][1], m[0][2] = s
    m[1][0], m[1][1], m[1][2] = u
    m[2][0], m[2][1], m[2][2] = (-f[0], -f[1], -f[2])
    m[3][3] = 1.0
    m[0][3] = - (s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2])
    m[1][3] = - (u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2])
    m[2][3] =   (f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2])
    return m

def flatten_col_major(m4):
    return [m4[0][0], m4[1][0], m4[2][0], m4[3][0],
            m4[0][1], m4[1][1], m4[2][1], m4[3][1],
            m4[0][2], m4[1][2], m4[2][2], m4[3][2],
            m4[0][3], m4[1][3], m4[2][3], m4[3][3]]

def flatten3x3(m3):
    return [m3[0][0], m3[1][0], m3[2][0],
            m3[0][1], m3[1][1], m3[2][1],
            m3[0][2], m3[1][2], m3[2][2]]

def mat3_from4(M):
    return [[M[0][0], M[0][1], M[0][2]],
            [M[1][0], M[1][1], M[1][2]],
            [M[2][0], M[2][1], M[2][2]]]

def transpose3(a):
    return [[a[0][0], a[1][0], a[2][0]],
            [a[0][1], a[1][1], a[2][1]],
            [a[0][2], a[1][2], a[2][2]]]

def inverse3(a):
    det = (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
         - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
         + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
    if abs(det) < 1e-12:
        return [[1,0,0],[0,1,0],[0,0,1]]
    invdet = 1.0/det
    return [[ (a[1][1]*a[2][2]-a[1][2]*a[2][1])*invdet,
              (a[0][2]*a[2][1]-a[0][1]*a[2][2])*invdet,
              (a[0][1]*a[1][2]-a[0][2]*a[1][1])*invdet ],
            [ (a[1][2]*a[2][0]-a[1][0]*a[2][2])*invdet,
              (a[0][0]*a[2][2]-a[0][2]*a[2][0])*invdet,
              (a[0][2]*a[1][0]-a[0][0]*a[1][2])*invdet ],
            [ (a[1][0]*a[2][1]-a[1][1]*a[2][0])*invdet,
              (a[0][1]*a[2][0]-a[0][0]*a[2][1])*invdet,
              (a[0][0]*a[1][1]-a[0][1]*a[1][0])*invdet ]]

def normal_matrix_from_model(M4):
    # inverse(transpose(M3))
    return inverse3(transpose3(mat3_from4(M4)))

def lerp(a, b, t): return a + (b - a) * t

def mat_lerp(mA, mB, t):
    out = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            out[r][c] = lerp(mA[r][c], mB[r][c], t)
    return out

# ---------- Case-insensitive path resolution ----------

def walk_case_insensitive(root: Path, relative: Path) -> Path | None:
    cur = root
    for part in relative.parts:
        try:
            entries = list(cur.iterdir())
        except Exception:
            return None
        lower = {e.name.lower(): e for e in entries}
        hit = lower.get(part.lower())
        if not hit:
            return None
        cur = hit
    return cur

def resolve_in_root_case_insensitive(root: Path, candidate: Path) -> Path | None:
    if candidate.is_absolute():
        try:
            rel = candidate.relative_to(Path('/'))
            return walk_case_insensitive(Path('/'), rel)
        except Exception:
            return candidate if candidate.exists() else None
    return walk_case_insensitive(root, candidate)

# ---------- Viewer ----------

def run_viewer(start_path: str | None = None, base_dir: str | None = None, trace: bool = False):
    import pyglet
    from pyglet import gl
    from pyglet.window import mouse, key
    from pyglet.graphics.shader import Shader, ShaderProgram

    # --- GL helpers ---
    def setup_texture_params(tex):
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex.id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        try: gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        except Exception: pass
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    # --- GLSL shaders (no leading whitespace/newline before #version!) ---
    vert_src = """#version 330 core
    in vec3 position;
    in vec2 tex_coords0;
    in vec3 normal;
    in vec3 tangent;

    uniform mat4 u_mvp;
    uniform mat4 u_model;   // <- use model, not u_normalMat

    out vec2 v_uv;
    out mat3 v_TBN;

    void main() {
        gl_Position = u_mvp * vec4(position, 1.0);
        v_uv = tex_coords0;

        // Build normal matrix from the upper-left 3x3 of u_model
        mat3 normalMat = transpose(inverse(mat3(u_model)));

        vec3 N = normalize(normalMat * normal);
        vec3 T = normalize(normalMat * tangent);
        T = normalize(T - dot(T, N) * N);
        vec3 B = cross(N, T);
        v_TBN = mat3(T, B, N);
    }
    """

    frag_src = """#version 330 core
    in vec2 v_uv;
    in mat3 v_TBN;

    uniform sampler2D u_tex;
    uniform sampler2D u_bump;
    uniform bool u_textured;
    uniform bool u_has_bump;
    uniform bool u_bump_is_height;

    uniform vec3 u_light_dir;  // world-space direction *toward* the light
    uniform vec3 u_light_col;
    uniform vec3 u_ambient;

    out vec4 fragColor;

    vec3 height_to_normal(vec2 uv) {
        ivec2 ts = textureSize(u_bump, 0);
        vec2 px = vec2(1.0) / vec2(ts);
        float hC = texture(u_bump, uv).r;
        float hR = texture(u_bump, uv + vec2(px.x, 0)).r;
        float hU = texture(u_bump, uv + vec2(0, px.y)).r;
        float scale = 0.5; // bump strength
        return normalize(vec3((hC - hR) * scale, (hC - hU) * scale, 1.0));
    }

    void main() {
        vec4 base = u_textured ? texture(u_tex, v_uv) : vec4(1.0);

        vec3 N_tan = vec3(0, 0, 1);
        if (u_has_bump) {
            if (u_bump_is_height) {
                N_tan = height_to_normal(v_uv);
            } else {
                vec3 m = texture(u_bump, v_uv).xyz * 2.0 - 1.0; // tangent-space normal map
                N_tan = normalize(m);
            }
        }

        vec3 N = normalize(v_TBN * N_tan);
        float ndl = max(dot(N, normalize(-u_light_dir)), 0.0);
        vec3 lit = u_ambient + u_light_col * ndl;

        fragColor = vec4(base.rgb * lit, base.a);
    }
    """

    program = ShaderProgram(Shader(vert_src, 'vertex'), Shader(frag_src, 'fragment'))
    program.use()
    program['u_light_dir'] = (-0.4, -0.9, -0.3)
    program['u_light_col'] = (0.9, 0.9, 0.9)
    program['u_ambient']   = (0.25, 0.25, 0.25)

    current_sod_path = [start_path]
    current_base_dir = [base_dir]
    loaded_sod = [None]  # keep parsed SOD for saving
    node_map = {}
    all_positions_world = []
    draw_items = []  # (tex_diffuse, tex_borg, tex_bump/height, bump_is_height, vlist, is_borg_branch, is_damage, node_name)

    texture_cache = {}
    tried_by_tex = defaultdict(list)
    resolved = {}

    # animation
    anim_play = False
    anim_time = 0.0
    anim_fps = 30.0
    anim_len_seconds = 0.0
    channels = {}

    # borg
    borg_branch_visible = True
    borg_tag_count = 0
    have_borg_root = [False]

    damage_visible = True
    damage_tag_count = 0

    # --- tree panel state ---
    panel_visible = True
    panel_w = 320
    row_h = 18
    tree_scroll = 0.0
    selected_name = None

    # tree data
    children = defaultdict(list)   # parent_name -> [child_names...]
    tree_roots = []                # top-level nodes
    expanded = {}                  # name -> bool (expanded/collapsed)
    hidden_nodes = set()           # names hidden (and their descendants)

    bg_panel = None  # will set after window is created

    # --- create window first ---
    window = pyglet.window.Window(1280, 720, caption="SOD Viewer", resizable=True)

    # now that window exists, build the background panel
    bg_panel = shapes.Rectangle(0, 0, panel_w, window.height, color=(20, 25, 32))
    bg_panel.opacity = 180
    gl.glClearColor(0.06, 0.08, 0.10, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    culling = True
    front_ccw = True
    gl.glEnable(gl.GL_CULL_FACE)

    label = pyglet.text.Label(
        "O: open | L: base | S: save as (1.8/1.93) | B: Borg | M dmg | A: anim | L-drag orbit | "
        "Scroll zoom | W wire | T tex | C cull | F flip | D dump | Space reset | Esc quit",
        font_size=12, x=10, y=10, anchor_x='left', anchor_y='bottom', color=(220,220,220,255)
    )
    hint = pyglet.text.Label(
        "Press O to open a .SOD  (or L to set a base texture folder)",
        font_size=16, x=window.width//2, y=window.height//2,
        anchor_x='center', anchor_y='center', color=(220,220,220,180)
    )

    wireframe = False
    textured = True
    yaw, pitch, dist = 45.0, 20.0, 10.0
    cx, cy, cz = 0.0, 0.0, 0.0

    white_img = pyglet.image.SolidColorImagePattern(color=(255,255,255,255)).create_image(2,2)
    white_tex = white_img.get_texture()
    normalflat_img = pyglet.image.SolidColorImagePattern(color=(128,128,255,255)).create_image(2,2)
    normalflat_tex = normalflat_img.get_texture()
    setup_texture_params(normalflat_tex)
    setup_texture_params(white_tex)

    # --------- texture resolving ---------
    def build_roots_for_base(base: Path, sod_dir: Path | None):
        roots = []
        if sod_dir: roots.append(sod_dir.resolve())
        def ci_sub(*parts):
            hit = walk_case_insensitive(base, Path(*parts))
            return hit if hit and hit.exists() else None
        r_sod = ci_sub('sod')
        r_tex = ci_sub('textures')
        r_rgb = ci_sub('textures', 'rgb')
        for r in (r_sod, r_tex, r_rgb):
            if r and r.is_dir(): roots.append(r.resolve())
        seen, ordered = set(), []
        for r in roots:
            if r not in seen: seen.add(r); ordered.append(r)
        return ordered

    def resolve_texture(texname: str, roots):
        if not texname: return None
        name = texname.replace("\\", "/")
        raw = Path(name)
        rels = [raw] if raw.suffix else [raw.with_suffix(".tga"), raw.with_suffix(".dds")]
        exts = {raw.suffix.lower()} if raw.suffix else {".tga", ".dds"}
        candidates = []
        for r in roots:
            for rel in rels:
                candidates.append((r, rel))
                candidates.append((r, Path(rel.name)))
        tried_by_tex[texname].clear()
        for root, rel in candidates:
            tried_by_tex[texname].append(str((root / rel)))
            hit = resolve_in_root_case_insensitive(root, rel)
            if hit and hit.exists():
                if hit.suffix.lower() in exts or not raw.suffix:
                    resolved[texname] = str(hit); return hit
        resolved[texname] = None; return None

    def resolve_borg_variant(texname: str, roots):
        if not texname:
            return None
        name = texname.replace("\\", "/")
        p = Path(name)
        if p.suffix:
            cand = p.with_name(p.stem + "_b" + p.suffix)
            hit = resolve_texture(str(cand), roots)
            if hit:
                return hit
        else:
            for ext in (".tga", ".dds"):
                cand = Path(p.name + "_b" + ext)
                hit = resolve_texture(str(cand), roots)
                if hit:
                    return hit
        return None

    def resolve_bump_variant(texname: str, roots):
        if not texname:
            return None
        p = Path(texname.replace("\\", "/"))
        bases = []
        if p.suffix:
            bases.append((p.stem + "_bump" + p.suffix))
        else:
            bases.extend([p.name + "_bump.tga", p.name + "_bump.dds"])
        for b in bases:
            hit = resolve_texture(b, roots)
            if hit: return hit
        return None

    # --------- borg helpers ---------
    def find_borg_root_name(nodes: dict[str, Node]) -> str | None:
        for k in nodes.keys():
            if k and k.lower() == 'borg': return k
        for k in nodes.keys():
            if k and 'borg' in k.lower(): return k
        return None

    def is_in_borg_branch(nodes: dict[str, Node], name: str, borg_root: str | None) -> bool:
        if not borg_root or not name or name not in nodes: return False
        target = borg_root.lower(); cur = name; seen=set()
        while cur and cur in nodes and cur not in seen:
            seen.add(cur)
            if cur.lower() == target: return True
            cur = nodes[cur].root
        return False

    def is_damage_node(name: str, node: Node) -> bool:
        """Heuristic: treat meshes whose name/material/texture suggests damage as 'damage' overlays."""
        toks = []
        if name: toks.append(name)
        if node and node.mesh:
            if node.mesh.texture: toks.append(node.mesh.texture)
            if node.mesh.material: toks.append(node.mesh.material)
            for g in node.mesh.groups:
                if g.material: toks.append(g.material)
        s = " ".join(toks).lower()
        # keep this conservative; expand if you find more patterns in the wild
        return ("damage" in s) or ("dmg" in s)

    # --------- animation helpers ---------
    def mat34_to_blender_local(mat34):
        return mat34_to_blender(mat34)

    def world_matrix_at(name: str, base_nodes: dict[str, Node], t: float) -> list[list[float]]:
        def local_mat_for(node_name: str):
            ch = channels.get(node_name)
            if ch and 'mats' in ch and ch['mats']:
                mats = ch['mats']
                if len(mats) == 1: return mat34_to_blender_local(mats[0])
                if anim_len_seconds <= 0.0: return mat34_to_blender_local(mats[0])
                frame = (t % anim_len_seconds) * anim_fps
                i0 = int(math.floor(frame)) % len(mats)
                i1 = (i0 + 1) % len(mats)
                alpha = frame - math.floor(frame)
                m0 = mat34_to_blender_local(mats[i0])
                m1 = mat34_to_blender_local(mats[i1])
                return mat_lerp(m0, m1, alpha)
            return mat34_to_blender_local(base_nodes[node_name].mat34)

        chain = []
        cur = name; seen=set()
        while cur and cur in base_nodes and cur not in seen:
            seen.add(cur); chain.append(cur); cur = base_nodes[cur].root or None
        M = local_mat_for(chain[-1])
        for n in reversed(chain[:-1]):
            M = mat_mul(M, local_mat_for(n))
        return M

    def rebuild_tree():
        nonlocal tree_roots
        children.clear()
        names = list(node_map.keys())

        # build children mapping
        for name, node in node_map.items():
            parent = node.root if node.root in node_map else None
            children[parent].append(name)

        # stable order
        for k in list(children.keys()):
            children[k].sort(key=lambda s: s.lower() if isinstance(s, str) else str(s))

        # collect top-level nodes
        tree_roots = children.get(None, [])[:]

        # default expanded state for new nodes
        for n in names:
            expanded.setdefault(n, True)

    def compute_rows():
        out = []
        def walk(n, d):
            out.append((n, d))
            if expanded.get(n, True):
                for c in children.get(n, []):
                    walk(c, d + 1)
        for r in tree_roots:
            walk(r, 0)
        return out

    def node_is_hidden(name: str) -> bool:
        """Hidden if this node or any ancestor is in hidden_nodes."""
        cur = name
        seen = set()
        while cur and cur in node_map and cur not in seen:
            if cur in hidden_nodes:
                return True
            seen.add(cur)
            cur = node_map[cur].root if node_map[cur].root in node_map else None
        return False

    # --------- scene (re)load ---------
    def load_scene(path: str):
        nonlocal all_positions_world, draw_items, yaw, pitch, dist, cx, cy, cz
        nonlocal channels, anim_len_seconds, node_map, borg_tag_count, have_borg_root, damage_tag_count
        draw_items.clear(); all_positions_world = []
        texture_cache.clear(); tried_by_tex.clear(); resolved.clear()
        borg_tag_count = 0; have_borg_root[0] = False
        channels = {}

        sod = SOD.from_file_path(path)
        loaded_sod[0] = sod
        node_map = sod.nodes
        channels = sod.channels
        damage_tag_count = 0

        hidden_nodes.clear()
        rebuild_tree()

        max_keys = 0
        for ch in channels.values():
            if 'mats' in ch: max_keys = max(max_keys, len(ch['mats']))
        anim_len_seconds = (max_keys / anim_fps) if max_keys > 0 else 0.0
        print(("[anim] detected %d key(s); ~%.0f FPS → %.3fs loop." % (max_keys, anim_fps, anim_len_seconds)) if max_keys>0 else "[anim] no transform keyframes.")

        sod_path = Path(path)
        base = Path(current_base_dir[0]) if current_base_dir[0] else sod_path.parent.parent
        current_base_dir[0] = str(base)
        tex_roots = build_roots_for_base(base, sod_path.parent)

        borg_root = find_borg_root_name(sod.nodes)
        if borg_root: have_borg_root[0] = True; print(f"[borg] root node detected: {borg_root}")
        else: print("[borg] no borg root found")

        unique_tex = []
        for node in sod.nodes.values():
            if node.mesh and (node.mesh.texture or "") not in unique_tex:
                unique_tex.append(node.mesh.texture or "")
        print("=== Textures referenced in SOD ===")
        for tname in unique_tex: print(" -", repr(tname))

        for name, node in sod.nodes.items():
            if not node.mesh:
                continue

            # base texture
            tex_path = resolve_texture(node.mesh.texture or "", tex_roots)
            if tex_path and tex_path not in texture_cache:
                try:
                    img = pyglet.image.load(str(tex_path))
                    texture_cache[tex_path] = img.get_texture()
                    setup_texture_params(texture_cache[tex_path])
                except Exception:
                    texture_cache[tex_path] = white_tex
            tex_normal = texture_cache.get(tex_path, white_tex)

            # borg variant
            borg_path = resolve_borg_variant(node.mesh.texture or "", tex_roots)
            tex_borg = None
            if borg_path:
                if borg_path not in texture_cache:
                    try:
                        img_b = pyglet.image.load(str(borg_path))
                        texture_cache[borg_path] = img_b.get_texture()
                        setup_texture_params(texture_cache[borg_path])
                    except Exception:
                        texture_cache[borg_path] = white_tex
                tex_borg = texture_cache.get(borg_path, white_tex)

            # bump/height map
            bump_path = None
            bump_is_height = False
            if node.mesh.bumpmap:
                bump_path = resolve_texture(node.mesh.bumpmap, tex_roots)
                bump_is_height = bool(node.mesh.use_heightmap)
            if not bump_path:
                bump_path = resolve_bump_variant(node.mesh.texture or "", tex_roots)
            tex_bump = None
            if bump_path:
                if bump_path not in texture_cache:
                    try:
                        img_h = pyglet.image.load(str(bump_path))
                        texture_cache[bump_path] = img_h.get_texture()
                        setup_texture_params(texture_cache[bump_path])
                    except Exception:
                        texture_cache[bump_path] = normalflat_tex
                tex_bump = texture_cache.get(bump_path, normalflat_tex)

            node_is_borg_branch = is_in_borg_branch(sod.nodes, name, borg_root)
            if node_is_borg_branch: borg_tag_count += 1
            
            is_damage = is_damage_node(name, node)
            if is_damage: damage_tag_count += 1            

            # build interleaved arrays, compute smooth normals & tangents
            local = [(-vx, vy, vz) for (vx,vy,vz) in node.mesh.verts]
            for g in node.mesh.groups:
                pos_flat, uv_flat = [], []

                for f in g.faces:
                    vs, uvs = [], []
                    for k in (0,1,2):
                        vi = f.indices[k]; ti = f.tc_indices[k]
                        p = local[vi]
                        t = node.mesh.tcs[ti] if node.mesh.tcs else (0.0, 0.0)
                        vs.append(p); uvs.append((t[0], 1.0 - t[1]))
                    for (px,py,pz) in vs:
                        pos_flat.extend([px,py,pz])
                    for (uu,vv) in uvs:
                        uv_flat.extend([uu,vv])

                if not pos_flat:
                    continue

                count = len(pos_flat)//3
                rawN = [(0.0,0.0,0.0)] * count
                rawT = [(0.0,0.0,0.0)] * count

                for i in range(0, count, 3):
                    p1 = (pos_flat[3*i+0], pos_flat[3*i+1], pos_flat[3*i+2])
                    p2 = (pos_flat[3*(i+1)+0], pos_flat[3*(i+1)+1], pos_flat[3*(i+1)+2])
                    p3 = (pos_flat[3*(i+2)+0], pos_flat[3*(i+2)+1], pos_flat[3*(i+2)+2])

                    uv1 = (uv_flat[2*i+0], uv_flat[2*i+1])
                    uv2 = (uv_flat[2*(i+1)+0], uv_flat[2*(i+1)+1])
                    uv3 = (uv_flat[2*(i+2)+0], uv_flat[2*(i+2)+1])

                    e1 = vec_sub(p2, p1); e2 = vec_sub(p3, p1)
                    duv1 = (uv2[0]-uv1[0], uv2[1]-uv1[1])
                    duv2 = (uv3[0]-uv1[0], uv3[1]-uv1[1])

                    n = vec_norm(vec_cross(e1, e2))

                    denom = (duv1[0]*duv2[1] - duv2[0]*duv1[1])
                    if abs(denom) < 1e-6:
                        t = (1.0, 0.0, 0.0)
                    else:
                        r = 1.0 / denom
                        t = ( (e1[0]*duv2[1]-e2[0]*duv1[1]) * r,
                              (e1[1]*duv2[1]-e2[1]*duv1[1]) * r,
                              (e1[2]*duv2[1]-e2[2]*duv1[1]) * r )
                        t = vec_norm(t)

                    rawN[i+0] = tuple(map(lambda a,b:a+b, rawN[i+0], n))
                    rawN[i+1] = tuple(map(lambda a,b:a+b, rawN[i+1], n))
                    rawN[i+2] = tuple(map(lambda a,b:a+b, rawN[i+2], n))
                    rawT[i+0] = tuple(map(lambda a,b:a+b, rawT[i+0], t))
                    rawT[i+1] = tuple(map(lambda a,b:a+b, rawT[i+1], t))
                    rawT[i+2] = tuple(map(lambda a,b:a+b, rawT[i+2], t))

                accN, accT = {}, {}
                for i in range(count):
                    key = (pos_flat[3*i+0], pos_flat[3*i+1], pos_flat[3*i+2],
                           uv_flat[2*i+0], uv_flat[2*i+1])
                    accN[key] = tuple(map(sum, zip(accN.get(key, (0,0,0)), rawN[i])))
                    accT[key] = tuple(map(sum, zip(accT.get(key, (0,0,0)), rawT[i])))

                norm_flat, tan_flat = [], []
                for i in range(count):
                    key = (pos_flat[3*i+0], pos_flat[3*i+1], pos_flat[3*i+2],
                           uv_flat[2*i+0], uv_flat[2*i+1])
                    n = vec_norm(accN[key]); t = vec_norm(accT[key])
                    t = vec_norm((t[0]-n[0]*vec_dot(t,n),
                                  t[1]-n[1]*vec_dot(t,n),
                                  t[2]-n[2]*vec_dot(t,n)))
                    norm_flat.extend(n); tan_flat.extend(t)

                vlist = program.vertex_list(
                    count, gl.GL_TRIANGLES,
                    position=('f3', pos_flat),
                    tex_coords0=('f2', uv_flat),
                    normal=('f3', norm_flat),
                    tangent=('f3', tan_flat),
                )

                draw_items.append((
                    tex_normal, tex_borg, tex_bump, bool(bump_path and bump_is_height),
                    vlist, node_is_borg_branch, is_damage, name
                ))

        for _texN, _texB, _texH, _is_height, vlist, _is_borg, _is_damage, node_name in draw_items:
            data = vlist.position
            M = world_matrix_at(node_name, node_map, 0.0)
            for i in range(0, len(data), 3):
                x,y,z = data[i], data[i+1], data[i+2]
                all_positions_world.append(transform_point(M, (x,y,z)))

        print("=== Texture resolution summary ===")
        for tname in unique_tex:
            print(f" • {tname!r} -> {resolved.get(tname)}")
            if resolved.get(tname) is None:
                for cand in tried_by_tex.get(tname, []):
                    print("    tried:", cand)

        if all_positions_world:
            (minx,miny,minz), (maxx,maxy,maxz) = compute_bounds(all_positions_world)
            cx, cy, cz = ((minx+maxx)/2.0, (miny+maxy)/2.0, (maxz+minz)/2.0)
            diag = math.sqrt((maxx-minx)**2 + (maxy-miny)**2 + (maxz-minz)**2)
            dist = max(2.0, diag * 1.5)

        if damage_tag_count > 0:
            print(f"[damage] tagged {damage_tag_count} group(s) as damage. Press M to toggle.")

        window.set_caption(f"SOD Viewer — {Path(path).name}")
        if have_borg_root[0]: print(f"[borg] tagged {borg_tag_count} group(s) as Borg branch. Press B to toggle.")
        if anim_len_seconds > 0: print("[anim] press A to play/pause.")

    # --------- saving ---------
    def write_identifier(f, name: str | None):
        if not name:
            f.write(struct.pack("<H", 0)); return
        data = name.encode('utf-8')
        f.write(struct.pack("<H", len(data)))
        f.write(struct.pack(f"<{len(data)}s", data))

    def save_as_sod(out_path: str, target_version: float):
        sod = loaded_sod[0]
        if not sod:
            print("[save] nothing loaded."); return
        if target_version not in (1.8, 1.93):
            print("[save] only 1.8 and 1.93 supported."); return

        with open(out_path, "wb") as f:
            f.write(b"Storm3D_SW")
            f.write(struct.pack("<f", target_version))

            mat_names = []
            seen = set()
            for n in sod.nodes.values():
                if n.mesh:
                    for m in [n.mesh.material] + [g.material for g in n.mesh.groups]:
                        m = m or "default"
                        if m not in seen:
                            seen.add(m); mat_names.append(m)
            f.write(struct.pack("<H", len(mat_names)))
            for name in mat_names:
                write_identifier(f, name)
                f.write(b"\x00" * (12*3 + 4 + 1))
                if target_version >= 1.9:
                    f.write(b"\x00")

            f.write(struct.pack("<H", len(sod.nodes)))
            for node in sod.nodes.values():
                f.write(struct.pack("<H", node.type))
                write_identifier(f, node.name or "")
                write_identifier(f, node.root or "")
                f.write(struct.pack("<12f", *node.mat34))
                if node.type == 12:
                    write_identifier(f, node.emitter or "")
                elif node.type == 1 and node.mesh:
                    mesh = node.mesh

                    if target_version >= 1.7:
                        write_identifier(f, mesh.material or "default")

                    if target_version >= 1.93:
                        mesh_flags = 4 if mesh.illumination else 0
                        f.write(struct.pack("<I", mesh_flags))
                        num_textures = 2 if mesh.bumpmap else 1
                        f.write(struct.pack("<I", num_textures))

                    write_identifier(f, mesh.texture or "")

                    if target_version == 1.91:
                        f.write(struct.pack("<H", 0))
                    elif target_version == 1.92:
                        write_identifier(f, mesh.assimilation_texture or "")
                        f.write(struct.pack("<H", 0))
                    elif target_version >= 1.93:
                        f.write(struct.pack("<I", 0))
                        if mesh.bumpmap:
                            write_identifier(f, mesh.bumpmap or "")
                            f.write(struct.pack("<I", 512))  # bump flag
                        write_identifier(f, mesh.assimilation_texture or "")
                        f.write(struct.pack("<H", 0))

                    f.write(struct.pack("<H", len(mesh.verts)))
                    f.write(struct.pack("<H", len(mesh.tcs)))
                    f.write(struct.pack("<H", len(mesh.groups)))
                    for (x,y,z) in mesh.verts:
                        f.write(struct.pack("<3f", x, y, z))
                    for (u,v) in mesh.tcs:
                        f.write(struct.pack("<2f", u, v))
                    for g in mesh.groups:
                        f.write(struct.pack("<H", len(g.faces)))
                        write_identifier(f, g.material or "default")
                        for face in g.faces:
                            for k in range(3):
                                f.write(struct.pack("<H", face.indices[k]))
                                f.write(struct.pack("<H", face.tc_indices[k]))
                    f.write(struct.pack("<b", mesh.cull_type if isinstance(mesh.cull_type, int) else 0))
                    f.write(struct.pack("<H", 0))

            anim_items = []
            for name, ch in sod.channels.items():
                if 'mats' in ch and ch['mats']:
                    anim_items.append((name, ch['atype'], 'mats', ch['mats']))
                elif 'scalars' in ch and ch['scalars']:
                    anim_items.append((name, ch['atype'], 'scalars', ch['scalars']))
            f.write(struct.pack("<H", len(anim_items)))
            for name, atype, kind, data in anim_items:
                write_identifier(f, name or "")
                kf = len(data)
                f.write(struct.pack("<H", kf))
                if kind == 'mats':
                    f.write(struct.pack("<I", 48 * kf))
                    f.write(struct.pack("<H", atype if isinstance(atype, int) else 0))
                    for m in data:
                        f.write(struct.pack("<12f", *m))
                else:
                    f.write(struct.pack("<I", 4 * kf))
                    f.write(struct.pack("<H", atype if isinstance(atype, int) else 5))
                    for s in data:
                        f.write(struct.pack("<f", float(s)))

            if target_version not in (1.4, 1.5):
                f.write(struct.pack("<H", 0))

        print(f"[save] wrote {out_path} (v{target_version:.2f})")

    # --------- clock update for animation time ---------
    def tick(dt):
        nonlocal anim_time
        if anim_play and anim_len_seconds > 0.0:
            anim_time = (anim_time + dt) % anim_len_seconds

    pyglet.clock.schedule_interval(tick, 1/120.0)

    @window.event
    def on_resize(width, height):
        nonlocal hint
        hint.x, hint.y = width//2, height//2
        if bg_panel:
            bg_panel.height = height
            bg_panel.width = panel_w
        return pyglet.event.EVENT_HANDLED

    @window.event
    def on_draw():
        window.clear()

        aspect = max(1e-3, window.width / float(window.height))
        P = mat_perspective(60.0, aspect, 0.1, 10000.0)
        rad_yaw, rad_pitch = math.radians(yaw), math.radians(pitch)
        eye = (cx + dist*math.cos(rad_pitch)*math.cos(rad_yaw),
               cy + dist*math.sin(rad_pitch),
               cz + dist*math.cos(rad_pitch)*math.sin(rad_yaw))
        V = mat_look_at(eye, (cx,cy,cz), (0,1,0))

        program.use()
        program['u_textured'] = bool(textured)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if wireframe else gl.GL_FILL)
        if culling: gl.glEnable(gl.GL_CULL_FACE)
        else:       gl.glDisable(gl.GL_CULL_FACE)
        gl.glFrontFace(gl.GL_CCW if front_ccw else gl.GL_CW)

        for texN, texB, texH, bump_is_height, v, is_borg_branch, is_damage, node_name in draw_items:
            if node_is_hidden(node_name):
                continue

            if is_damage and (not damage_visible):
                continue
            
            if is_borg_branch and (not borg_branch_visible):
                continue

            # Build matrices
            M   = world_matrix_at(node_name, node_map, anim_time if anim_play else 0.0)
            MVP = mat_mul(P, mat_mul(V, M))

            # Upload them
            program['u_mvp']   = (ctypes.c_float * 16)(*flatten_col_major(MVP))
            program['u_model'] = (ctypes.c_float * 16)(*flatten_col_major(M))

            # base (or borg) texture on unit 0
            chosen = (texB if (borg_branch_visible and texB) else texN)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, chosen.id if textured else white_tex.id)
            program['u_tex'] = 0

            # bump/height on unit 1 (if present)
            has_bump = bool(texH)
            program['u_has_bump'] = has_bump
            program['u_bump_is_height'] = bool(bump_is_height)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, (texH or normalflat_tex).id)
            program['u_bump'] = 1

            v.draw(gl.GL_TRIANGLES)

        # unbind
        gl.glActiveTexture(gl.GL_TEXTURE1); gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # --- draw tree panel + UI overlay ---
        if panel_visible and bg_panel:
            # UI pass: draw solid, no culling, no depth
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_DEPTH_TEST)

            bg_panel.width = panel_w
            bg_panel.height = window.height
            bg_panel.draw()

            rows = compute_rows()
            y_top = window.height - 8
            for i, (name, depth) in enumerate(rows):
                y = y_top - i * row_h + tree_scroll
                if y < 4 or y > window.height - 4:
                    continue
                has_kids = bool(children.get(name))
                expander = "▾" if expanded.get(name, True) else "▸"
                expander = expander if has_kids else "•"
                vis = "[x]" if not node_is_hidden(name) else "[ ]"
                is_sel = (name == selected_name)
                col = (255, 255, 160, 255) if is_sel else (220, 220, 220, 255)

                x_base = 8 + depth * 14
                pyglet.text.Label(expander, x=x_base, y=y, anchor_y='center', color=col, font_size=12).draw()
                pyglet.text.Label(vis, x=x_base + 16, y=y, anchor_y='center', color=col, font_size=12).draw()
                pyglet.text.Label(name, x=x_base + 44, y=y, anchor_y='center', color=col, font_size=12).draw()

            gl.glEnable(gl.GL_DEPTH_TEST)  # restore for next 3D pass (next frame)

        # Always draw HUD text in solid fill, with no culling/depth:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        label.draw()
        if not draw_items:
            hint.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        nonlocal yaw, pitch
        if buttons & mouse.LEFT:
            yaw += dx * 0.3
            pitch += dy * 0.3
            pitch = max(-89.9, min(89.9, pitch))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        nonlocal dist, tree_scroll
        if panel_visible and x <= panel_w:
            # scroll tree: wheel up -> move content down slightly (tweak multiplier as you like)
            rows = compute_rows()
            content_h = max(0, len(rows) * row_h)
            view_h = window.height
            max_up = 0.0  # top-most
            max_down = max(0.0, content_h - view_h + 16)  # allow a bit of slack

            tree_scroll += scroll_y * (row_h * 3)
            # clamp so you can't scroll crazy far
            if tree_scroll > max_up:
                tree_scroll = max_up
            if tree_scroll < -max_down:
                tree_scroll = -max_down
            return  # don't zoom camera when scrolling the panel

        # original camera zoom
        factor = 0.9 if scroll_y > 0 else 1.1
        dist = max(0.1, dist * factor)

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal wireframe, textured, yaw, pitch, dist, culling, front_ccw
        nonlocal anim_play, anim_time, borg_branch_visible
        nonlocal panel_visible, selected_name, hidden_nodes, expanded, tree_roots
        nonlocal damage_visible 
        from pyglet.window import key

        if symbol == key.W:
            wireframe = not wireframe
        elif symbol == key.T:
            textured = not textured
        elif symbol == key.C:
            culling = not culling
        elif symbol == key.F:
            front_ccw = not front_ccw

        elif symbol == key.M:
            damage_visible = not damage_visible
            print(f"[damage] meshes are now {'VISIBLE' if damage_visible else 'HIDDEN'}")
                
        elif symbol == key.H:
            panel_visible = not panel_visible
        elif symbol == key.V:
            if selected_name:
                if selected_name in hidden_nodes:
                    hidden_nodes.remove(selected_name)
                else:
                    hidden_nodes.add(selected_name)
        elif symbol == key.E:
            # Expand/collapse all
            if any(not expanded.get(n, True) for n in node_map):
                for n in node_map:
                    expanded[n] = True
            else:
                for n in node_map:
                    expanded[n] = False
                # Keep roots visible so you don't "lose" the tree completely
                for r in roots:
                    expanded[r] = True
                       
        elif symbol == key.D:
            print("=== Texture resolution summary (D) ===")
            for t in sorted(set(resolved.keys())):
                print(f" • {t!r} -> {resolved.get(t)}")
                if resolved.get(t) is None:
                    for cand in tried_by_tex.get(t, []):
                        print("    tried:", cand)
        elif symbol == key.B:
            if not have_borg_root[0]:
                print("[borg] toggle: no 'borg' node detected in this file.")
            else:
                borg_branch_visible = not borg_branch_visible
                print(f"[borg] branch is now {'VISIBLE' if borg_branch_visible else 'HIDDEN'}")
        elif symbol == key.A:
            if anim_len_seconds <= 0.0:
                print("[anim] no transform keyframes present.")
            else:
                anim_play = not anim_play
                print("[anim] ▶ play" if anim_play else "[anim] ⏸ pause")
        elif symbol == key.S:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk(); root.withdraw()

                default_v = 1.93 if (loaded_sod[0] and loaded_sod[0].version >= 1.9) else 1.8
                ftypes = (
                    [("Star Trek Armada II [1.93]", "*.sod"),
                     ("Star Trek Armada [1.8]", "*.sod"),
                     ("All files", "*.*")]
                    if default_v == 1.93 else
                    [("Star Trek Armada [1.8]", "*.sod"),
                     ("Star Trek Armada II [1.93]", "*.sod"),
                     ("All files", "*.*")]
                )

                typevar = tk.StringVar(value=ftypes[0][0])
                initialdir = None
                if current_sod_path[0]:
                    try:
                        from pathlib import Path as _P
                        initialdir = str(_P(current_sod_path[0]).parent)
                    except Exception:
                        pass

                fname = filedialog.asksaveasfilename(
                    title="Save SOD",
                    defaultextension=".sod",
                    filetypes=ftypes,
                    typevariable=typevar,
                    initialdir=initialdir
                )
                root.destroy()
                if not fname:
                    return

                label = typevar.get() or ftypes[0][0]
                tgt = 1.93 if "1.93" in label else 1.8
                save_as_sod(fname, tgt)
            except Exception as e:
                print("Save As failed:", e)

        elif symbol == key.SPACE:
            yaw, pitch = 45.0, 20.0
        elif symbol == key.O:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk(); root.withdraw()
                file = filedialog.askopenfilename(
                    title="Open SOD",
                    filetypes=[("SOD files", "*.SOD *.sod"), ("All files", "*.*")],
                )
                root.destroy()
                if file:
                    current_sod_path[0] = file
                    current_base_dir[0] = str(Path(file).parent.parent)
                    anim_time = 0.0; anim_play = False
                    load_scene(file)
            except Exception as e:
                print("Open dialog failed:", e)
        elif symbol == key.L:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk(); root.withdraw()
                folder = filedialog.askdirectory(title="Pick base texture folder (parent of 'sod' & 'textures')")
                root.destroy()
                if folder:
                    current_base_dir[0] = folder
                    if current_sod_path[0]:
                        anim_time = 0.0; anim_play = False
                        load_scene(current_sod_path[0])
            except Exception as e:
                print("Folder dialog failed:", e)
        elif symbol == key.ESCAPE:
            window.close()

    @window.event
    def on_mouse_press(x, y, buttons, modifiers):
        nonlocal selected_name
        if not panel_visible or x > panel_w:
            return
        rows = compute_rows()
        y_top = window.height - 8
        idx = int((y_top - y + tree_scroll) // row_h)
        if idx < 0 or idx >= len(rows):
            return

        name, depth = rows[idx]
        has_kids = bool(children.get(name))
        x_base = 8 + depth * 14

        within_expander = (x >= x_base - 2) and (x <= x_base + 12)
        within_checkbox = (x >= x_base + 16) and (x <= x_base + 16 + 24)

        if has_kids and within_expander:
            expanded[name] = not expanded.get(name, True)
        elif within_checkbox:
            if name in hidden_nodes:
                hidden_nodes.remove(name)
            else:
                hidden_nodes.add(name)
        else:
            selected_name = name

    @window.event
    def on_close():
        pyglet.app.exit()
        
    # start loop after all @window.event handlers are defined:
    pyglet.app.run()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="SOD Viewer (pyglet 2.x) — textures, Borg toggle, animation, save-as")
    ap.add_argument("sod_path", nargs="?", help="Optional path to a .SOD file to load on startup")
    ap.add_argument("--tex-root", help="Optional base dir (parent of 'sod' & 'textures')", default=None)
    ap.add_argument("--trace-textures", action="store_true", help="Print detailed texture lookup attempts to console")
    args = ap.parse_args()

    try:
        import pyglet  # noqa: F401
    except Exception:
        print("Needs pyglet. Install with:  python -m pip install --user pyglet")
        sys.exit(1)

    run_viewer(args.sod_path, args.tex_root, args.trace_textures)

if __name__ == "__main__":
    main()
