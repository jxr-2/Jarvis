import sounddevice as sd
import numpy as np
import whisper
import wave
import os
import tempfile
import time
# IMPORTANTE USAR PYTHON 3-10 o 3.9
# Configuración de audio
DURATION = 5  # segundos por fragmento de audio
SAMPLERATE = 16000
CHANNELS = 1

# Cargar modelo de Whisper (elige: tiny, base, small, medium, large)
model_name = input("Modelo Whisper (tiny, base, small, medium, large) [base]: ") or "base"
print("Aclaración: Entre mas grande el modelo, más lento será el reconocimiento de voz.")
if model_name not in ["tiny", "base", "small", "medium", "large"]:
    print("❌ Modelo no válido. Usando 'base' por defecto.")
    model_name = "base"
print(f"Cargando modelo Whisper '{model_name}'...")
model = whisper.load_model(model_name)
print("Modelo cargado.")

def grabar_audio_temp(nombre_archivo):
    print("🎙️ Grabando...")
    audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS, dtype='int16')
    sd.wait()  # Esperar a que termine
    print("✅ Grabación finalizada.")

    # Guardar como archivo WAV temporal
    with wave.open(nombre_archivo, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16 bits = 2 bytes
        wf.setframerate(SAMPLERATE)
        wf.writeframes(audio.tobytes())

def transcribir_y_borrar(nombre_archivo):
    print("🧠 Transcribiendo...")
    try:
        resultado = model.transcribe(nombre_archivo)
        print("🗣️ Texto:", resultado["text"])
    except Exception as e:
        print(f"❌ Error en la transcripción: {e}")
    finally:
        os.remove(nombre_archivo)
        print("🧹 Archivo temporal eliminado.\n")

def bucle_principal():
    try:
        while True:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                grabar_audio_temp(temp_audio.name)
                transcribir_y_borrar(temp_audio.name)
            time.sleep(1)  # Pausa de 1 segundo entre grabaciones
    except KeyboardInterrupt:
        print("🛑 Salida del programa.")

if __name__ == "__main__":
    bucle_principal()
