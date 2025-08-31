from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordRequestForm
from app.security.auth import authenticate_user, create_access_token, get_current_user
import uvicorn

app = FastAPI(title="Contextual AI Orchestrator", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = authenticate_user(form_data.username, form_data.password)
    if not username:
        return {"error": "Invalid credentials"}
    token = create_access_token({"sub": username})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/protected")
def protected_route(current_user: str = Depends(get_current_user)):
    return {"message": f"Hello, {current_user}. You have access to this protected route!"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)
