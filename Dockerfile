# Todo Dockerfile "começa" de algum lugar, podendo ser uma versão do Ubuntu, 
# ou do python, ou até uma customizada do DockerHub. É com o "FROM" que você define isso.
# Usamos uma imagem base do Python
FROM python:3.9

# Cria um diretório de trabalho
WORKDIR /app

# Copiamos os arquivos de onde está o Dockerfile para o container
COPY . .

# Instalamos as dependências descritas no requirements.txt
RUN pip install -r requirements.txt

# Definimos o comando que será executado ao rodar o container
CMD ["python3", "app.py"]