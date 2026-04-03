import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
import os

# --- Configuration ---
# Folder containing parasite_1.xlsx, parasite_2.xlsx, ... files
SCENE_FOLDER = 'Folder7_non-abortive_parasite_features'   # ← change to your scene folder

# ── FPS — auto-read from video, falls back to manual value ───────────────────
VIDEO_PATH   = 'test_videos/Folder7_non-abortive.mp4'     # ← change to match your scene
FPS_FALLBACK = 30.0                          # used if video file is not found

def get_fps(video_path, fallback):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                print(f"✅ FPS read from video: {fps:.3f}")
                return fps
    except Exception:
        pass
    print(f"⚠️  Could not read video — using fallback FPS = {fallback}")
    return fallback

# ── PELT tuning ──────────────────────────────────────────────────────────────
PEN      = 50    # penalty — raise for fewer CPs, lower for more sensitivity
MIN_SIZE = 50    # minimum frames between two consecutive change points

# ── Smoothing windows ─────────────────────────────────────────────────────────
AREA_SMOOTH_WINDOW = 5   # Area chart  — raise for smoother, 1 = no smoothing
DA_SMOOTH_WINDOW   = 15   # dA/dt chart — raise for smoother, 1 = no smoothing


def detect_area_expansion(file_path, fps=None, output_image_path=None, output_text_path=None):
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}.")
        return

    # ── Get FPS ──────────────────────────────────────────────────────────────
    if fps is None:
        fps = get_fps(VIDEO_PATH, FPS_FALLBACK)

    print("Loading data...")
    df = pd.read_excel(file_path)

    for col in ('Frame_Index', 'Area'):
        if col not in df.columns:
            print(f"Error: column '{col}' not found. "
                  f"Available: {df.columns.tolist()}")
            return

    df = df.sort_values('Frame_Index').reset_index(drop=True)
    frame_indices = df['Frame_Index'].values
    area          = df['Area'].values

    # Helper: frames → seconds
    def to_sec(frames):
        return frames / fps

    # ── 1. Smooth area ────────────────────────────────────────────────────────
    smooth_window = AREA_SMOOTH_WINDOW
    area_smooth   = (pd.Series(area)
                       .rolling(smooth_window, center=True, min_periods=1)
                       .median()
                       .values)

    # ── 2. Area velocity dA/dt ────────────────────────────────────────────────
    dA = np.gradient(area_smooth, frame_indices)   # px² / frame

    # Smooth dA/dt — rolling median removes segmentation spike noise
    dA_smooth_window = DA_SMOOTH_WINDOW
    dA_smooth = (pd.Series(dA)
                   .rolling(dA_smooth_window, center=True, min_periods=1)
                   .median()
                   .values)

    # ── 3. Standardize ────────────────────────────────────────────────────────
    X        = np.column_stack([area_smooth, dA_smooth])
    X_scaled = StandardScaler().fit_transform(X)

    # ── 4. PELT ───────────────────────────────────────────────────────────────
    print("Running PELT on Area signal ...")
    algo       = rpt.Pelt(model="rbf", min_size=MIN_SIZE).fit(X_scaled)
    cp_indices = algo.predict(pen=PEN)

    detected_frames = [frame_indices[i - 1]
                       for i in cp_indices
                       if i < len(frame_indices)]
    detected_times  = [to_sec(f) for f in detected_frames]

    # ── 5. Phase table ────────────────────────────────────────────────────────
    all_boundaries = [frame_indices[0]] + detected_frames + [frame_indices[-1]]
    phases = []
    for k in range(len(all_boundaries) - 1):
        start_fr  = all_boundaries[k]
        end_fr    = all_boundaries[k + 1]
        area_s    = area_smooth[np.searchsorted(frame_indices, start_fr)]
        area_e    = area_smooth[np.searchsorted(frame_indices, end_fr,
                                                 side='right') - 1]
        delta     = area_e - area_s
        dur_fr    = end_fr - start_fr
        dur_sec   = to_sec(dur_fr)
        label     = ("↑ EXPANSION" if delta > 0
                     else "↓ SHRINK"    if delta < 0
                     else "→ STABLE")
        seg_mask   = (frame_indices >= start_fr) & (frame_indices <= end_fr)
        seg_f      = frame_indices[seg_mask].astype(float)
        seg_a      = area_smooth[seg_mask]
        mean_slope = float(np.polyfit(seg_f, seg_a, 1)[0]) if len(seg_f) >= 2 else 0.0
        phases.append({
            'phase'          : k + 1,
            'start_frame'    : start_fr,
            'end_frame'      : end_fr,
            'start_time_s'   : round(to_sec(start_fr), 2),
            'end_time_s'     : round(to_sec(end_fr),   2),
            'duration_frames': dur_fr,
            'duration_sec'   : round(dur_sec, 2),
            'area_start'     : area_s,
            'area_end'       : area_e,
            'delta_area'     : delta,
            'mean_slope'     : round(mean_slope, 3),
            'label'          : label,
        })

    # ── 6. Expansion onset calculation ──────────────────────────────────────
    # Find which phase contains the global area maximum
    global_max_frame = frame_indices[np.argmax(area_smooth)]
    max_phase_idx    = next(i for i, p in enumerate(phases)
                            if p['start_frame'] <= global_max_frame <= p['end_frame'])

    # All phases strictly before the max phase
    # Segments before the max phase + the max phase itself
    pre_max_phases = phases[:max_phase_idx]

    # Always skip the first segment — it may be noisy regardless of direction
    if pre_max_phases:
        pre_max_phases = pre_max_phases[1:]

    # Find the first positive-slope segment among the remaining pre-max phases
    first_pos_idx = next((i for i, p in enumerate(pre_max_phases)
                          if p['mean_slope'] > 0), None)

    max_phase = phases[max_phase_idx]

    if first_pos_idx is not None:
        # Pre-max positive segments + max phase always included regardless of slope
        expansion_phases  = pre_max_phases[first_pos_idx:] + [max_phase]
        expansion_frames  = sum(p['duration_frames'] for p in expansion_phases)
        expansion_sec     = sum(p['duration_sec']    for p in expansion_phases)
        expansion_ph_nums = [p['phase'] for p in expansion_phases]
    else:
        # No positive pre-max segments — count max phase alone
        expansion_phases  = [max_phase]
        expansion_frames  = max_phase['duration_frames']
        expansion_sec     = max_phase['duration_sec']
        expansion_ph_nums = [max_phase['phase']]

    # ── 7. Console summary ────────────────────────────────────────────────────
    print(f"\n{'═'*84}")
    print(f"  AREA-BASED EXPANSION ANALYSIS  |  FPS = {fps:.3f}")
    print(f"{'═'*84}")
    print(f"  Change points  →  frames: {detected_frames}")
    print(f"                 →  times:  {[f'{t:.2f}s' for t in detected_times]}")
    print()
    hdr = (f"  {'Ph':<4} {'St.Fr':>7} {'En.Fr':>7} {'St.s':>7} {'En.s':>7} "
           f"{'Dur(fr)':>8} {'Dur(s)':>7} {'ΔArea':>10} {'Slope(px²/fr)':>14}  Label")
    print(hdr)
    print(f"  {'─'*84}")
    for p in phases:
        marker = ' ◀ expansion' if p['phase'] in expansion_ph_nums else ''
        print(f"  {p['phase']:<4} {p['start_frame']:>7} {p['end_frame']:>7} "
              f"{p['start_time_s']:>7.2f} {p['end_time_s']:>7.2f} "
              f"{p['duration_frames']:>8} {p['duration_sec']:>7.2f} "
              f"{p['delta_area']:>10.0f} {p['mean_slope']:>14.3f}  {p['label']}{marker}")
    print(f"  {'─'*84}")
    if expansion_ph_nums:
        print(f"  ▶ EXPANSION ONSET  : phases {expansion_ph_nums}")
        print(f"                       {expansion_frames} frames  →  {expansion_sec:.2f} seconds")
        print(f"                       from {phases[expansion_ph_nums[0]-1]['start_time_s']:.2f}s ")
        print(f"                       to   {phases[max_phase_idx]['end_time_s']:.2f}s  (peak @ {global_max_frame/fps:.2f}s)")
    else:
        print(f"  ▶ EXPANSION ONSET  : no positive-slope segments found before maximum")
    print(f"{'═'*84}\n")

    # ── 8. Save text report ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_text_path) or '.', exist_ok=True)
    with open(output_text_path, 'w') as f:
        f.write(f"Area-Based Expansion Change Points  |  FPS = {fps:.3f}\n")
        f.write("=" * 84 + "\n")
        f.write(f"Change-point frames : {detected_frames}\n")
        f.write(f"Change-point times  : {[f'{t:.2f}s' for t in detected_times]}\n\n")
        f.write(f"{'Ph':<4} {'St.Fr':>7} {'En.Fr':>7} {'St.s':>7} {'En.s':>7} "
                f"{'Dur(fr)':>8} {'Dur(s)':>7} {'ΔArea':>10} {'Slope(px²/fr)':>14}  Label\n")
        f.write("-" * 84 + "\n")
        for p in phases:
            marker = ' <- expansion' if p['phase'] in expansion_ph_nums else ''
            f.write(f"{p['phase']:<4} {p['start_frame']:>7} {p['end_frame']:>7} "
                    f"{p['start_time_s']:>7.2f} {p['end_time_s']:>7.2f} "
                    f"{p['duration_frames']:>8} {p['duration_sec']:>7.2f} "
                    f"{p['delta_area']:>10.0f} {p['mean_slope']:>14.3f}  {p['label']}{marker}\n")
        f.write("\n" + "=" * 84 + "\n")
        if expansion_ph_nums:
            f.write(f"EXPANSION ONSET SUMMARY\n")
            f.write(f"  Phases counted : {expansion_ph_nums}\n")
            f.write(f"  Total frames   : {expansion_frames}\n")
            f.write(f"  Total seconds  : {expansion_sec:.2f} s\n")
            f.write(f"  Start time     : {phases[expansion_ph_nums[0]-1]['start_time_s']:.2f} s\n")
            f.write(f"  Peak reached   : {global_max_frame/fps:.2f} s\n")
        else:
            f.write("EXPANSION ONSET SUMMARY\n")
            f.write("  No positive-slope segments found before maximum.\n")
    print(f"✅ Saved text report  → {output_text_path}")

    # ── 8. Plot ───────────────────────────────────────────────────────────────
    time_indices = frame_indices / fps

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f'Parasite Area Expansion — PELT  |  FPS = {fps:.1f}',
        fontsize=15, weight='bold')

    # Panel 1 : Area
    ax1 = axes[0]
    ax1.plot(time_indices, area,        color='lightsteelblue',
             linewidth=1, alpha=0.6,   label='Area (raw)')
    ax1.plot(time_indices, area_smooth, color='steelblue',
             linewidth=2,              label=f'Area (smoothed, w={smooth_window})')
    ax1.set_ylabel('Area (px²)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    phase_colors = {'↑ EXPANSION': '#d4f1d4',
                    '↓ SHRINK'   : '#fdd5d5',
                    '→ STABLE'   : '#f5f5e0'}
    y_max = area_smooth.max()
    for p in phases:
        t_start = p['start_time_s']
        t_end   = p['end_time_s']
        ax1.axvspan(t_start, t_end, alpha=0.25,
                    color=phase_colors.get(p['label'], 'grey'))
        mid = (t_start + t_end) / 2
        is_expansion_ph = p['phase'] in expansion_ph_nums
        slope_color = 'darkgreen' if p['mean_slope'] > 0 else 'crimson' if p['mean_slope'] < 0 else '#555'
        border_lw   = 2.0 if is_expansion_ph else 0
        # Phase annotation: duration + label at top, slope below
        ax1.text(mid, y_max,
                 f"{p['duration_sec']:.1f}s\n({p['duration_frames']}fr)\n"
                 f"{p['label'].split()[0]}",
                 ha='center', va='top', fontsize=7.5, color='#333333',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5,
                           ec='#228B22' if is_expansion_ph else 'none', lw=border_lw))
        # Draw linear regression line over the segment (skip the max-containing phase)
        seg_mask   = (frame_indices >= p['start_frame']) & (frame_indices <= p['end_frame'])
        seg_t      = time_indices[seg_mask]
        seg_a      = area_smooth[seg_mask]
        if len(seg_t) >= 2 and p['phase'] != max_phase['phase']:
            coeffs   = np.polyfit(seg_t, seg_a, 1)
            reg_line = np.polyval(coeffs, seg_t)
            ax1.plot(seg_t, reg_line, color=slope_color,
                     linewidth=2, linestyle='--', alpha=0.85, zorder=4)
        # Mean slope label at vertical midpoint of the regression line
        y_mid_seg = float(np.median(seg_a)) if seg_mask.sum() > 0 else y_max * 0.5
        ax1.text(mid, y_mid_seg,
                 f"slope\n{p['mean_slope']:+.1f}",
                 ha='center', va='center', fontsize=7, color=slope_color,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec=slope_color, lw=0.8))

    for cp_t in detected_times:
        ax1.axvline(x=cp_t, color='red', linestyle='--', linewidth=1.8, alpha=0.85)
        ax1.text(cp_t, ax1.get_ylim()[0],
                 f' {cp_t:.1f}s', color='red', fontsize=7, va='bottom')

    # ── Global max & min markers on smoothed area ─────────────────────────────
    global_max_idx = np.argmax(area_smooth)
    global_min_idx = np.argmin(area_smooth)

    ax1.scatter(time_indices[global_max_idx], area_smooth[global_max_idx],
                color='darkgreen', s=80, zorder=5)
    ax1.annotate(f"MAX\n{area_smooth[global_max_idx]:.0f} px²\n"
                 f"@ {time_indices[global_max_idx]:.1f}s",
                 xy=(time_indices[global_max_idx], area_smooth[global_max_idx]),
                 xytext=(10, -35), textcoords='offset points',
                 fontsize=8, color='darkgreen',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2))

    ax1.scatter(time_indices[global_min_idx], area_smooth[global_min_idx],
                color='darkred', s=80, zorder=5)
    ax1.annotate(f"MIN\n{area_smooth[global_min_idx]:.0f} px²\n"
                 f"@ {time_indices[global_min_idx]:.1f}s",
                 xy=(time_indices[global_min_idx], area_smooth[global_min_idx]),
                 xytext=(10, 15), textcoords='offset points',
                 fontsize=8, color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2))

    # Panel 2 : dA/dt
    ax2 = axes[1]
    ax2.plot(time_indices, dA, color='darkorange', linewidth=1, alpha=0.25,
             label='dA/dt (raw)')
    ax2.plot(time_indices, dA_smooth, color='darkorange', linewidth=2,
             label=f'dA/dt (smoothed, w={dA_smooth_window})')
    ax2.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    ax2.set_ylabel('dA/dt  (px²/frame)', fontsize=12)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    for cp_t in detected_times:
        ax2.axvline(x=cp_t, color='red', linestyle='--', linewidth=1.8, alpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
    plt.savefig(output_image_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved chart         → {output_image_path}")


# ── Run — auto-discover all parasite_*.xlsx files in SCENE_FOLDER ───────────
import glob, re

def run_all(scene_folder):
    pattern = os.path.join(scene_folder, 'parasite_*.xlsx')
    files   = sorted(glob.glob(pattern),
                     key=lambda f: int(re.search(r'parasite_(\d+)', f).group(1)))

    if not files:
        print(f"❌  No parasite_*.xlsx files found in '{scene_folder}'")
        return

    print(f"\n{'═'*62}")
    print(f"  Found {len(files)} parasite file(s) in '{scene_folder}'")
    for f in files:
        print(f"    • {os.path.basename(f)}")
    print(f"{'═'*62}\n")

    fps = get_fps(VIDEO_PATH, FPS_FALLBACK)

    for excel_path in files:
        # Build per-parasite output paths inside the same scene folder
        basename = os.path.splitext(os.path.basename(excel_path))[0]  # e.g. parasite_2
        img_path  = os.path.join(scene_folder, f'{basename}_Area_Expansion_Chart.png')
        txt_path  = os.path.join(scene_folder, f'{basename}_Detected_Expansion_Points.txt')

        print(f"\n{'─'*62}")
        print(f"  Processing: {basename}")
        print(f"{'─'*62}")
        detect_area_expansion(excel_path, fps=fps,
                              output_image_path=img_path,
                              output_text_path=txt_path)

    print(f"\n✅  All done — {len(files)} parasite(s) processed.")

run_all(SCENE_FOLDER)
