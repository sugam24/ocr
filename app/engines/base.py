from abc import ABC, abstractmethod
from typing import Dict, Any

class OCRModel(ABC):
    """
    Abstract interface for OCR models to ensure the HTTP layer remains decoupled
    from the specific inference backend (HuggingFace, vLLM, etc.).
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the model weights and initialize resources. 
        This should be called exactly once at startup.
        """
        pass

    @abstractmethod
    async def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Run inference on the provided image bytes.
        
        Args:
            image_data: Raw bytes of the image file.

        Returns:
            A dictionary containing the OCR results:
            {
                "text": str,          # Extracted text content
                "blocks": list,       # Reserved for future structured output
                "model_version": str  # Model identifier
            }
        """
        pass
