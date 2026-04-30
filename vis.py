import numpy as np
import tifffile
import pyvista as pv
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image as PILImage

# ── Глобальные данные ─────────────────────────────────────────────────────────

data = {}          # image, mask, colors, z_positions, focal, base_dist
pl_state = {"pl": None, "open": False}
_syncing  = False

# ── Загрузка данных ───────────────────────────────────────────────────────────

def load_data():
    img_path  = Path(img_path_var.get())
    mask_path = Path(mask_path_var.get())
    xy_um     = float(xy_um_var.get())
    z_um      = float(z_um_var.get())
    gap_um    = float(gap_var.get())

    if not img_path.exists():
        messagebox.showerror("Ошибка", f"Файл не найден:\n{img_path}"); return False
    if not mask_path.exists():
        messagebox.showerror("Ошибка", f"Файл не найден:\n{mask_path}"); return False

    image = tifffile.imread(img_path)
    mask  = tifffile.imread(mask_path)
    nz, ny, nx = image.shape

    if mask.shape != image.shape:
        tmp = np.zeros(image.shape, dtype=mask.dtype)
        s = tuple(slice(0, min(a,b)) for a,b in zip(mask.shape, image.shape))
        tmp[s] = mask[s]; mask = tmp

    vmax   = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else float(image.max())
    labels = np.unique(mask); labels = labels[labels != 0]
    pal    = plt.colormaps["tab20"].resampled(max(len(labels), 1))
    colors = {lbl: np.array(pal(i % 20)[:3]) for i, lbl in enumerate(labels)}
    z_pos  = np.arange(nz) * (z_um + gap_um)
    focal  = np.array([nx*xy_um/2, ny*xy_um/2, nz*(z_um+gap_um)/2])
    bdist  = np.linalg.norm([nx*xy_um, ny*xy_um, nz*(z_um+gap_um)]) * 1.2

    data.update(dict(image=image, mask=mask, nz=nz, ny=ny, nx=nx,
                     vmax=vmax, labels=labels, colors=colors,
                     z_positions=z_pos, focal=focal, base_dist=bdist,
                     xy_um=xy_um, z_um=z_um, gap_um=gap_um))

    # предвычисляем меши один раз
    status.config(text="Вычисляю меши..."); root.update()
    meshes = {}
    for lbl in labels:
        vol = (mask == lbl).astype(np.uint8)
        if vol.sum() < 10: continue
        try:
            vol_pad = np.pad(vol, ((1,1),(1,1),(1,1)), mode="constant")
            verts, faces, *_ = marching_cubes(vol_pad, level=0.5)
        except Exception: continue
        meshes[lbl] = (verts, faces)
    data["meshes"] = meshes

    status.config(text=f"Загружено: {image.shape}  instances: {len(labels)}  меши: {len(meshes)}")
    return True

# ── Вычисление камеры ─────────────────────────────────────────────────────────

def camera_from_angles(rx_deg, rz_deg, dist_mult):
    focal     = data["focal"]
    base_dist = data["base_dist"]
    theta = np.radians(90 - rx_deg)
    phi   = np.radians(rz_deg)
    dist  = base_dist * dist_mult
    dx = dist * np.sin(theta) * np.cos(phi)
    dy = dist * np.sin(theta) * np.sin(phi)
    dz = dist * np.cos(theta)
    cam_pos  = focal + np.array([dx, dy, dz])
    view_dir = focal - cam_pos; view_dir /= np.linalg.norm(view_dir)
    g_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(view_dir, g_up)
    if np.linalg.norm(right) < 1e-6:
        g_up = np.array([0.0, 1.0, 0.0]); right = np.cross(view_dir, g_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view_dir); up /= np.linalg.norm(up)
    return tuple(cam_pos), tuple(focal), tuple(up)

def angles_from_camera(pl):
    cam = np.array(pl.camera.position)
    foc = np.array(pl.camera.focal_point)
    d   = cam - foc; dist = np.linalg.norm(d)
    if dist < 1e-6: return 70.0, -40.0, 1.0
    theta = np.arccos(np.clip(d[2]/dist, -1, 1))
    rx    = 90.0 - np.degrees(theta)
    rz    = np.degrees(np.arctan2(d[1], d[0]))
    return rx, np.clip(rz, -180, 180), np.clip(dist/data["base_dist"], 0.3, 4.0)

def apply_camera(pl):
    pos, foc, up = camera_from_angles(rx_var.get(), rz_var.get(), dist_var.get())
    pl.camera.position = pos; pl.camera.focal_point = foc; pl.camera.up = up

# ── Построение сцены ──────────────────────────────────────────────────────────

def build_scene(pl, show_img=True):
    d = data
    nz, ny, nx   = d["nz"], d["ny"], d["nx"]
    xy_um, z_um  = d["xy_um"], d["z_um"]
    gap_um       = d["gap_um"]
    z_positions  = d["z_positions"]
    image, mask  = d["image"], d["mask"]
    vmax         = d["vmax"]
    colors       = d["colors"]
    labels       = d["labels"]

    if show_img:
     for z in range(nz):
        z_pos = z_positions[z]
        sl  = (image[z].astype(np.float64) / vmax * 255).astype(np.uint8)
        rgb = np.stack([sl]*3, axis=-1).copy()
        for lbl, c in colors.items():
            px = mask[z] == lbl
            if px.any():
                c8 = (c*255).astype(np.uint8)
                rgb[px] = (0.35*rgb[px] + 0.65*c8).astype(np.uint8)
        plane = pv.Plane(
            center=(nx*xy_um/2, ny*xy_um/2, z_pos), direction=(0,0,1),
            i_size=nx*xy_um, j_size=ny*xy_um, i_resolution=1, j_resolution=1)
        actor = pl.add_mesh(plane, texture=pv.Texture(rgb[::-1]), opacity=0.95, lighting=False)
        pl._slice_actors = getattr(pl, '_slice_actors', [])
        pl._slice_actors.append(actor)
        x1, y1 = nx*xy_um, ny*xy_um
        corners = [(0,0,z_pos),(x1,0,z_pos),(x1,y1,z_pos),(0,y1,z_pos)]
        for i in range(4):
            a = pl.add_mesh(pv.Line(corners[i], corners[(i+1)%4]),
                        color="white", line_width=0.8, opacity=0.35)
            pl._slice_actors.append(a)

    meshes = d["meshes"]
    for lbl, (verts, faces) in meshes.items():
        vx = (verts[:,2] - 1) * xy_um
        vy = (verts[:,1] - 1) * xy_um
        z_ext = np.concatenate([[z_positions[0]-(z_um+gap_um)], z_positions, [z_positions[-1]+(z_um+gap_um)]])
        vz  = np.interp(verts[:,0], np.arange(nz+2), z_ext)
        pts = np.column_stack([vx, vy, vz])
        fc  = np.column_stack([np.full(len(faces),3), faces]).astype(np.int_)
        surf = pv.PolyData(pts, fc).smooth(n_iter=50, relaxation_factor=0.1)
        pl.add_mesh(surf, color=colors[lbl], opacity=0.92,
                    smooth_shading=True, show_edges=False, specular=0.5, specular_power=30)

    pl.remove_all_lights()
    for pos, intensity in [
        ((nx*xy_um*2,  ny*xy_um*2,  nz*(z_um+gap_um)*3), 0.7),
        ((-nx*xy_um,   ny*xy_um/2,  nz*(z_um+gap_um)*2), 0.4),
        ((nx*xy_um/2, -ny*xy_um,    nz*(z_um+gap_um)*1), 0.25),
    ]:
        light = pv.Light(position=pos, light_type="scene light")
        light.intensity = intensity; pl.add_light(light)
    pl.set_background([0.05, 0.05, 0.07])
    pl.enable_anti_aliasing("ssaa")
    pl.enable_ssao(radius=15, bias=0.5, kernel_size=128)  # ambient occlusion — затемняет углы/стыки
    pl.enable_eye_dome_lighting()                          # контурная подсветка по глубине

# ── GUI ───────────────────────────────────────────────────────────────────────

root = tk.Tk()
root.title("3D Render — Settings")
root.resizable(False, False)
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()

nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True, padx=8, pady=8)

# ────────────────── Tab 1: Файлы ──────────────────────────────────────────────
tab_files = ttk.Frame(nb, padding=14)
nb.add(tab_files, text="  Файлы & Пиксели  ")

def make_file_row(parent, row, label, default, filetypes, save=False):
    ttk.Label(parent, text=label, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
    var = tk.StringVar(value=default)
    ent = ttk.Entry(parent, textvariable=var, width=52)
    ent.grid(row=row, column=1, padx=6, pady=4)
    def browse():
        if save:
            p = filedialog.asksaveasfilename(defaultextension=".png",
                    filetypes=filetypes, initialfile=Path(var.get()).name)
        else:
            p = filedialog.askopenfilename(filetypes=filetypes,
                    initialdir=str(Path(var.get()).parent))
        if p: var.set(p)
    ttk.Button(parent, text="…", width=3, command=browse).grid(row=row, column=2, pady=4)
    return var

tif_ft  = [("TIFF files", "*.tif *.tiff"), ("All", "*.*")]
png_ft  = [("PNG", "*.png")]

img_path_var  = make_file_row(tab_files, 0, "Изображение (.tif)",
    "C:/Users/podtyazhki/Desktop/PHD/bias/20260209_160932_pixsize.ome.tif", tif_ft)
mask_path_var = make_file_row(tab_files, 1, "Маска (.tif)",
    "C:/Users/podtyazhki/Desktop/PHD/bias/20260209_160932_m_8bit.ome.tif", tif_ft)
out_path_var  = make_file_row(tab_files, 2, "Сохранить PNG",
    "C:/Users/podtyazhki/Desktop/PHD/bias/graphical_abstract.png", png_ft, save=True)

ttk.Separator(tab_files, orient="horizontal").grid(
    row=3, column=0, columnspan=3, sticky="ew", pady=10)

ttk.Label(tab_files, text="Размер пикселя", font=("Segoe UI", 10, "bold")).grid(
    row=4, column=0, columnspan=3, sticky="w", pady=(0,6))

def make_px_field(parent, row, label, default):
    ttk.Label(parent, text=label, width=22, anchor="e").grid(row=row, column=0, sticky="e", padx=(0,6), pady=4)
    var = tk.StringVar(value=str(default))
    ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="w", pady=4)
    ttk.Label(parent, text="µm").grid(row=row, column=2, sticky="w")
    return var

xy_um_var = make_px_field(tab_files, 5, "XY размер пикселя:", "0.439")
z_um_var  = make_px_field(tab_files, 6, "Z шаг (µm/слайс):", "1.0")

# ────────────────── Tab 2: Параметры рендера ──────────────────────────────────
tab_rend = ttk.Frame(nb, padding=14)
nb.add(tab_rend, text="  Рендер  ")

# реестр всех entry-виджетов слайдеров: var → entry
_slider_entries = {}

def make_slider_entry(parent, row, label, from_, to, default, resolution=0.1):
    """Слайдер + поле ввода. Значение применяется только по Enter."""
    ttk.Label(parent, text=label, width=24, anchor="e").grid(
        row=row, column=0, padx=(0,8), pady=6, sticky="e")
    var = tk.DoubleVar(value=default)

    sl = ttk.Scale(parent, from_=from_, to=to, variable=var,
                   orient="horizontal", length=220)
    sl.grid(row=row, column=1, pady=6)

    entry_var = tk.StringVar(value=f"{default:.2f}")
    ent = ttk.Entry(parent, textvariable=entry_var, width=8)
    ent.grid(row=row, column=2, padx=(8,0))

    # слайдер → поле (только если поле не в фокусе)
    def _sl_to_ent(*_):
        if root.focus_get() is not ent:
            entry_var.set(f"{var.get():.2f}")
    var.trace_add("write", _sl_to_ent)

    # поле → слайдер только по Enter
    def _apply(event=None):
        try:
            v = float(entry_var.get())
            v = max(from_, min(to, v))
            var.set(v)
            ent.config(foreground="black")
        except ValueError:
            ent.config(foreground="red")
    ent.bind("<Return>", _apply)

    # подсвечиваем поле пока редактируется
    ent.bind("<Key>", lambda e: ent.config(foreground="#e87000"))

    _slider_entries[id(var)] = ent
    return var

ttk.Label(tab_rend, text="Камера", font=("Segoe UI", 10, "bold")).grid(
    row=0, column=0, columnspan=3, sticky="w", pady=(0,6))

rx_var   = make_slider_entry(tab_rend, 1, "Наклон вокруг X (°)",    0,  90,  70)
rz_var   = make_slider_entry(tab_rend, 2, "Вращение вокруг Z (°)", -180, 180, -40)
dist_var = make_slider_entry(tab_rend, 3, "Расстояние (×)",         0.3,  4,  1.0)

ttk.Separator(tab_rend, orient="horizontal").grid(
    row=4, column=0, columnspan=3, sticky="ew", pady=10)

ttk.Label(tab_rend, text="Стек", font=("Segoe UI", 10, "bold")).grid(
    row=5, column=0, columnspan=3, sticky="w", pady=(0,6))

gap_var = make_slider_entry(tab_rend, 6, "Зазор между слайсами (µm)", 0, 20, 3.0)

ttk.Separator(tab_rend, orient="horizontal").grid(
    row=7, column=0, columnspan=3, sticky="ew", pady=10)

res_var = tk.StringVar(value="1920x1080")
ttk.Label(tab_rend, text="Разрешение PNG", width=24, anchor="e").grid(
    row=9, column=0, padx=(0,8), sticky="e")
ttk.Combobox(tab_rend, textvariable=res_var, width=14, state="readonly",
    values=["1280x720","1920x1080","2560x1440","3840x2160"]).grid(
    row=9, column=1, sticky="w", pady=6)

show_image_var  = tk.BooleanVar(value=True)
transp_bg_var   = tk.BooleanVar(value=False)

def toggle_image_visibility(*_):
    pl = pl_state["pl"]
    if pl is None or not pl_state["open"]: return
    vis = show_image_var.get()
    for actor in getattr(pl, "_slice_actors", []):
        actor.SetVisibility(vis)
    pl.render()

ttk.Checkbutton(tab_rend, text="Показывать изображение", variable=show_image_var,
                command=toggle_image_visibility).grid(
    row=10, column=0, columnspan=3, sticky="w", padx=(80,0), pady=3)
ttk.Checkbutton(tab_rend, text="Прозрачный фон при сохранении", variable=transp_bg_var).grid(
    row=11, column=0, columnspan=3, sticky="w", padx=(80,0), pady=3)

# ── Статус + кнопки ───────────────────────────────────────────────────────────

bot = ttk.Frame(root, padding=(8,0,8,8))
bot.pack(fill="x")

status = ttk.Label(bot, text="Выбери файлы и нажми Open 3D",
                   foreground="#888", font=("Segoe UI", 9))
status.pack(side="bottom", pady=(6,0))

btn_row = ttk.Frame(bot)
btn_row.pack()

def do_open():
    status.config(text="Загрузка данных..."); root.update()
    if not load_data(): return
    if pl_state["pl"] is not None:
        try: pl_state["pl"].close()
        except Exception: pass
    pl = pv.Plotter(window_size=[screen_w, screen_h])
    build_scene(pl)
    pl.add_axes(line_width=3)
    pl.add_camera_orientation_widget()
    apply_camera(pl)
    pl.show(interactive_update=True, auto_close=False)
    pl_state["pl"] = pl; pl_state["open"] = True
    status.config(text="Двигай слайдеры — рендер обновляется в реальном времени")

def do_save():
    if not data:
        messagebox.showwarning("Нет данных", "Сначала нажми Open 3D"); return
    out    = Path(out_path_var.get())
    transp = transp_bg_var.get()
    w, h   = [int(x) for x in res_var.get().split("x")]

    status.config(text=f"Рендеринг {w}×{h}..."); root.update()

    # всегда рендерим в отдельный off-screen плоттер с нужным разрешением
    show_img = show_image_var.get()
    bg_col = [0, 0, 0]  # чёрный фон всегда — для маскировки прозрачности
    pl2 = pv.Plotter(off_screen=True, window_size=[w, h])
    pl2.set_background([0, 0, 0])
    build_scene(pl2, show_img=show_img)
    apply_camera(pl2)
    img = pl2.screenshot(None, return_img=True)
    pl2.close()

    if transp:
        # фон чёрный [0,0,0] → пиксели где RGB < 8 по всем каналам = фон
        is_bg = (img[:,:,:3].astype(np.int16).max(axis=2) < 8)
        alpha = np.where(is_bg, 0, 255).astype(np.uint8)
        PILImage.fromarray(np.dstack([img[:,:,:3], alpha]), "RGBA").save(str(out))
    else:
        # пересохраняем с правильным фоном
        pl3 = pv.Plotter(off_screen=True, window_size=[w, h])
        pl3.set_background([0.05, 0.05, 0.07])
        build_scene(pl3, show_img=show_img)
        apply_camera(pl3)
        pl3.screenshot(str(out))
        pl3.close()

    status.config(text=f"Сохранено → {out.name}  ({w}×{h})")
    print(f"Saved → {out}")

ttk.Button(btn_row, text="▶  Open 3D",   width=16, command=do_open).pack(side="left", padx=6, pady=4)
ttk.Button(btn_row, text="💾  Save PNG", width=16, command=do_save).pack(side="left", padx=6, pady=4)

# ── Реал-тайм обновление камеры + синхронизация слайдеров ────────────────────

def on_camera_change(*_):
    global _syncing
    if _syncing: return
    pl = pl_state["pl"]
    if pl is None or not pl_state["open"] or not data: return
    try: apply_camera(pl); pl.render()
    except Exception: pl_state["open"] = False

for v in (rx_var, rz_var, dist_var):
    v.trace_add("write", on_camera_change)

def on_gap_change(*_):
    """Зазор изменился — перестроить сцену."""
    if not data or not pl_state["open"]: return
    status.config(text="Перестраиваю сцену..."); root.update()
    try:
        load_data()
        pl = pl_state["pl"]
        pl.clear()
        build_scene(pl)
        apply_camera(pl)
        pl.render()
        status.config(text="Готово")
    except Exception as e:
        status.config(text=f"Ошибка: {e}")

gap_var.trace_add("write", lambda *_: root.after(400, on_gap_change))

def poll():
    global _syncing
    pl = pl_state["pl"]
    if pl is not None and pl_state["open"] and data:
        try:
            pl.update()
            rx, rz, dm = angles_from_camera(pl)
            focused = root.focus_get()
            _syncing = True
            # обновляем только те переменные, чьё поле НЕ в фокусе
            if focused is not _slider_entries.get(id(rx_var)):
                rx_var.set(round(rx, 1))
            if focused is not _slider_entries.get(id(rz_var)):
                rz_var.set(round(rz, 1))
            if focused is not _slider_entries.get(id(dist_var)):
                dist_var.set(round(dm, 3))
            _syncing = False
        except Exception:
            _syncing = False; pl_state["open"] = False
    root.after(30, poll)

poll()
root.mainloop()