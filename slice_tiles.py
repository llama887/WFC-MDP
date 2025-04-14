from PIL import Image
import os

image_path = "vectoraith_tileset_terrain_grassland_B_48x48.png" 
tile_size = 48
output_dir = "tiles_48x48_B"  

tileset = Image.open(image_path)
img_width, img_height = tileset.size

cols = img_width // tile_size
rows = img_height // tile_size

os.makedirs(output_dir, exist_ok=True)

print(f"Slicing {rows} rows x {cols} columns from '{image_path}'...")

for row in range(rows):
    for col in range(cols):
        left = col * tile_size
        top = row * tile_size
        right = left + tile_size
        bottom = top + tile_size

        tile = tileset.crop((left, top, right, bottom))
        filename = f"tile_{row}_{col}.png"
        tile.save(os.path.join(output_dir, filename))

print(f"{rows * cols} tiles saved to '{output_dir}/'")
