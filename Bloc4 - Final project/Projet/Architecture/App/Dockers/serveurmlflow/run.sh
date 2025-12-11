# Load environment variables from .env file (in parent directory)
Get-Content ../.env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2].Trim('"')
        Set-Item -Path "env:$name" -Value $value
    }
}

# lancement du serveur ML Flow
docker run -it -p 4000:4000 -v "$(pwd):/home/app" -e PORT=4000 -e AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY -e BACKEND_STORE_URI=$env:BACKEND_STORE_URI -e ARTIFACT_ROOT=$env:ARTIFACT_STORE_URI bloc03_mlflow
docker run -it -p 8501:8501 -e BACKEND_STORE_URI=$env:BACKEND_STORE_URI -e ARTIFACT_ROOT=$env:ARTIFACT_STORE_URI -v "$(pwd):/home/app" bloc03_streamlit
# BLOC 03 #
# pour lancer l'application streamlit qui permet de visualiser le tableau de bord des transactions de la veille 
# docker run -it -p 6000:6000 -v "$(pwd):/home/app" -e PORT=6000 -e AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY -e BACKEND_STORE_URI=$env:BACKEND_STORE_URI -e ARTIFACT_ROOT=$env:ARTIFACT_ROOT api_prediction