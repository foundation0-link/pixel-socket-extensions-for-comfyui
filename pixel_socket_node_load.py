import base64
import io
import numpy as np

from PIL import Image
from comfy_api.latest import io as comfy_api_io # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
import httpx

from .pixel_socket_utils import PixelSocketUtils

class PixelSocketLoadImageFromUrlNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketLoadImageFromUrlNode",
            display_name="Load Image From URL Node",
            category="PixelSocket/Load",
            is_output_node=True,
            inputs=[
                comfy_api_io.String.Input("image_url",
                    default="",
                    multiline=False,
                    optional=False
                )
            ],
            outputs=[
                comfy_api_io.Image.Output("image"),
            ]
        )

    @classmethod
    def execute(cls, image_url: str, **kwargs) -> None:
        try:
            img_data: bytes = b""
            if image_url.startswith("data:image/"):
                _, encoded = image_url.split(",", 1)
                img_data = base64.b64decode(encoded)

            elif image_url.startswith("http://") or image_url.startswith("https://"):
                response = httpx.get(image_url)
                response.raise_for_status()
                img_data = response.content
            else:
                print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Unsupported URL scheme.")

            # Validate image data
            if not img_data or not cls._validate_image_data(img_data):
                print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Invalid image data. Returning blank 1024x1024 image.")
                return PixelSocketUtils.create_fallback_image()

            img = Image.open(io.BytesIO(img_data)).convert("RGBA")

            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            return comfy_api_io.NodeOutput(img_tensor)

        except Exception as ex:
            print(f"[PixelSocketLoadImageFromUrlNode] ERROR: {ex}")
            import traceback
            traceback.print_exc()

        return PixelSocketUtils.create_fallback_image()

    @classmethod
    def _validate_image_data(cls, img_data: bytes) -> bool:
        """画像データが適切であるか判定"""
        MAX_SIZE = 10 * 1024 * 1024  # 10MB

        # ファイルサイズチェック
        if len(img_data) > MAX_SIZE:
            print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Image size {len(img_data)} bytes exceeds 10MB limit")
            return False

        # 画像フォーマットチェック（マジックナンバー）
        if len(img_data) < 4:
            print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Image data too small: {len(img_data)} bytes")
            return False

        # 既知の画像フォーマットのマジックナンバーをチェック
        magic_numbers = [
            (b'\x89PNG', 'PNG'),       # PNG
            (b'\xff\xd8\xff', 'JPEG'), # JPEG
            (b'GIF8', 'GIF'),          # GIF
            (b'RIFF', 'WebP'),         # WebP (RIFF format)
        ]

        is_valid_format = False
        for magic, fmt in magic_numbers:
            if img_data.startswith(magic):
                print(f"[PixelSocketLoadImageFromUrlNode] Valid {fmt} image detected ({len(img_data)} bytes)")
                is_valid_format = True
                break

        if not is_valid_format:
            print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Unknown or unsupported image format")
            return False

        # PIL で開けるかテスト
        try:
            Image.open(io.BytesIO(img_data)).verify()
            return True
        except Exception as e:
            print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Image verification failed: {e}")
            return False
