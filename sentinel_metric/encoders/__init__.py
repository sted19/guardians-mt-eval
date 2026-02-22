from .info_xlm import InfoXLMEncoder
from .xlmr import XLMREncoder

str2encoder = {
    "XLM-RoBERTa": XLMREncoder,
    "InfoXLM": InfoXLMEncoder,
}
