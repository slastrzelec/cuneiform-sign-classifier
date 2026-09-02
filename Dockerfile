# Obraz Dockera dla demo Streamlit klasyfikatora znakow klinowych.
#
# Celowo NIE pakujemy calego data/processed (train/val to tysiace obrazow,
# potrzebne tylko do treningu, nie do dzialania demo) - obraz zawiera
# jedynie to, co potrzebne do uruchomienia aplikacji: kod, wytrenowany
# model i galerie przykladow testowych do wyboru w interfejsie.

FROM python:3.11-slim

WORKDIR /app

# Zaleznosci najpierw (osobna warstwa cache'owana przez Dockera -
# nie przebudowuje sie przy kazdej zmianie kodu, tylko przy zmianie requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod aplikacji
COPY src/ src/
COPY train.py .
COPY app.py .

# Wytrenowany model (tylko najlepszy checkpoint, bez last_model.pt
# ktory zawiera dodatkowo stan optymalizatora - niepotrzebny do inferencji)
COPY checkpoints/best_model.pt checkpoints/best_model.pt

# Galeria przykladow do demo (tylko test/, nie train/val - te sluza
# wylacznie do treningu i nie sa potrzebne w dzialajacej aplikacji)
COPY data/processed/test/ data/processed/test/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
