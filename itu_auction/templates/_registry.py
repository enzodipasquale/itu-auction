from .TU import TU_template
from .LU import LU_template
from .convex_tax import convex_tax_template


TEMPLATE_REGISTRY = {
                    "TU": TU_template,
                    "LU": LU_template,  
                    "convex_tax": convex_tax_template,
                    }

def get_template(name):
    try:
        return TEMPLATE_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown template name: {name}")
