from comfy_api.latest import ComfyExtension, io as comfy_api_io # pyright: ignore[reportMissingImports]

class PixelSocketExtensions(ComfyExtension):
    async def get_node_list(self) -> list[type[comfy_api_io.ComfyNode]]:
        from .pixel_socket_delivery import PixelSocketDeliveryImageNode
        from .pixel_socket_load_pnginfo import PixelSocketLoadImageInfoNode
        from .pixel_socket_resize import PixelSocketResizeImageNode
        return [
                    PixelSocketDeliveryImageNode,
                    PixelSocketLoadImageInfoNode,
                    PixelSocketResizeImageNode,
               ]

async def comfy_entrypoint() -> ComfyExtension:
    return PixelSocketExtensions()
