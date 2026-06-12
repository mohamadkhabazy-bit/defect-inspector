# %%
import numpy as np
import cv2


class DefectHeatmap:
    def __init__(self, width: int = 200, height: int = 200):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.float32)

    def update(self, centers: list, img_w: int, img_h: int):
        for cx, cy in centers:
            gx = int(cx / img_w * self.width)
            gy = int(cy / img_h * self.height)

            gx = np.clip(gx, 0, self.width - 1)
            gy = np.clip(gy, 0, self.height - 1)

            radius = 8

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy

                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        dist = np.sqrt(dx**2 + dy**2)
                        self.grid[ny, nx] += np.exp(-dist**2 / (2 * (radius / 2)**2))

    def render(self, output_size: tuple = (400, 400)) -> np.ndarray:
        if self.grid.max() == 0:
            blank = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            cv2.putText(blank, "No detections yet",
                        (30, output_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            return blank

        norm = cv2.normalize(self.grid, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        resized = cv2.resize(norm, output_size, interpolation=cv2.INTER_CUBIC)

        heatmap_bgr = cv2.applyColorMap(resized, cv2.COLORMAP_JET)

        step = output_size[0] // 10
        for i in range(0, output_size[0], step):
            cv2.line(heatmap_bgr, (i, 0), (i, output_size[1]), (30, 30, 30), 1)
            cv2.line(heatmap_bgr, (0, i), (output_size[0], i), (30, 30, 30), 1)

        return heatmap_bgr

    def render_rgb(self, output_size: tuple = (400, 400)) -> np.ndarray:
        bgr = self.render(output_size)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

# %%



