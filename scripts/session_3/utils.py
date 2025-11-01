from fastapi import APIRouter

utils_router = APIRouter(prefix="/utils")

@utils_router.get("/health1")
def health_check(dump_input: int):
    if dump_input > 10:
        return { "message" : f"dump_input larger than 10: input value: {dump_input}" }
    else:
        return { "message" : f"dump_input less than 10: input value: {dump_input}" }
    
@utils_router.get("/health2")
def health_check(dump_input: int):
    if dump_input > 20:
        return { "message" : f"dump_input larger than 20: input value: {dump_input}" }
    else:
        return { "message" : f"dump_input less than 20: input value: {dump_input}" }