import numpy as np

from PIL import Image
from comfy_api.latest import io as comfy_api_io # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]

from .pixel_socket_utils import PixelSocketUtils

class PixelSocketResizeImageNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketResizeImageNode",
            display_name="Resize Image Node",
            category="PixelSocket/Processing",
            is_output_node=True,
            inputs=[
                comfy_api_io.Image.Input("image"),
                comfy_api_io.Int.Input("width",
                    default=1024,
                    min=0,
                    step=8,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Int.Input("height",
                    default=1024,
                    min=0,
                    step=8,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
            ],
            outputs=[
                comfy_api_io.Image.Output("image"),
                comfy_api_io.Int.Output("width"),
                comfy_api_io.Int.Output("height"),
            ]
        )

    @classmethod
    def execute(cls, image: torch.Tensor, width: int, height: int, **kwargs) -> None:
        try:
            img = PixelSocketUtils.tensor_to_image(image)
            img = img.convert("RGBA")

            # アスペクト比を維持しながらwidth/height以内の最大サイズにリサイズ
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height

            if width / height > aspect_ratio:
                # 高さに合わせる
                new_height = height
                new_width = int(height * aspect_ratio)
            else:
                # 幅に合わせる
                new_width = width
                new_height = int(width / aspect_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width_output, height_output = img.size

            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            return comfy_api_io.NodeOutput(img_tensor, width_output, height_output)

        except Exception as ex:
            print(f"[PixelSocketResizeImageNode] ERROR: {ex}")
            import traceback
            traceback.print_exc()

        return PixelSocketUtils.create_fallback_image(width, height)
