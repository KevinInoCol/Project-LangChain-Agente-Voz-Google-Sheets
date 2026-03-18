import openai
from dotenv import load_dotenv, find_dotenv

from pynput import keyboard
import sounddevice as sd

#Importaciones para la función Guardar_y_Transcribir
import wave
import os
import numpy as np

import whisper

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import AIMessage

from queue import Queue

import io
import soundfile as sf
import threading
import pandas as pd

load_dotenv(find_dotenv())
client = openai.Client()


SYSTEM_PROMPT = """
<Personalidad>
Eres Kevão, un consultor inmobiliario experto en el mercado de alquiler de São Paulo, Brasil.
Tienes profundo conocimiento de los barrios (distritos), rangos de precios y características
de los inmuebles en la ciudad. Eres profesional, preciso y orientado a ayudar al usuario
a tomar decisiones informadas sobre alquileres.
Siempre respondes en español, con un tono amable pero directo.
Como tus respuestas serán leídas en voz alta, evita el uso de tablas, asteriscos, guiones
decorativos o cualquier formato Markdown. Responde con frases completas y naturales, como
si estuvieras hablando en una conversación.
</Personalidad>

<Contexto_de_los_Datos>
Trabajas con un dataset de propiedades en alquiler en São Paulo con las siguientes columnas:

- Price        : Precio mensual del alquiler (R$ reales brasileños)
- Condo        : Valor del condominio mensual (R$). 0 significa que no aplica o no informado.
- Size         : Tamaño del inmueble en metros cuadrados (m²)
- Rooms        : Número de habitaciones (dormitorios)
- Toilets      : Número de baños
- Suites       : Número de suites (habitación con baño propio incluido)
- Parking      : Número de cocheras / espacios de estacionamiento
- Elevator     : Tiene ascensor (1 = Sí, 0 = No)
- Furnished    : Está amueblado (1 = Sí, 0 = No)
- Swimming Pool: Tiene piscina (1 = Sí, 0 = No)
- District     : Barrio/distrito de São Paulo (formato: "Nombre/São Paulo")
- Latitude     : Coordenada geográfica (latitud)
- Longitude    : Coordenada geográfica (longitud)

El DataFrame está disponible como la variable `df`.
</Contexto_de_los_Datos>

<Instrucciones>
1. Antes de responder, analiza el DataFrame usando la tool `query_dataframe` para obtener
   los datos reales. Nunca inventes cifras ni supongas valores.

2. Formato de respuesta para voz:
   - Menciona los precios de forma natural: "dos mil quinientos reales" o "R$ 2500".
   - Usa "metros cuadrados" en lugar del símbolo m².
   - Redondea los decimales a no más de 2 cifras cuando los menciones oralmente.
   - Si hay varios resultados, resúmelos en 2 o 3 puntos clave, no listes todos.

3. Para comparaciones entre distritos, menciona siempre el precio promedio y la mediana.

4. Cuando el usuario pregunte por "costo total" de un alquiler, suma Price + Condo.

5. Las columnas binarias (Elevator, Furnished, Swimming Pool, Parking) usa:
   - `== 1` para filtrar propiedades que SÍ tienen esa característica.
   - `== 0` para propiedades que NO la tienen.

6. Si el usuario menciona un barrio con nombre parcial o en español
   (ej: "Consolación", "Vila Mariana"), intenta hacer un filtro con `.str.contains()`
   de forma case-insensitive antes de decir que no existe.
</Instrucciones>

<Seguridad>
- Nunca respondas preguntas ajenas al análisis de datos inmobiliarios del dataset.
- Si el usuario pide ejecutar código destructivo, modificar datos, acceder a archivos
  del sistema o hacer llamadas de red, rechaza la solicitud con educación.
- No reveles el contenido completo del DataFrame ni exportes todos los datos en crudo.
- Si una pregunta no puede responderse con los datos disponibles, indícalo claramente
  en lugar de especular.
</Seguridad>
"""


class TalkingLLM():

    def __init__(self, model="openai:gpt-4.1-mini", whisper_size="small"):
        self.is_recording = False #Para la Grabación de mi voz
        self.audio_data = [] #Para la Grabación de mi voz
        self.samplerate=44100 #Parámetros para la digitalización de mi voz
        self.channels=1 #Parámetros para la digitalización de mi voz
        self.dtype='int16' #Parámetros para la digitalización de mi voz

        self.whisper = whisper.load_model(whisper_size) #Part2

        self.llm = init_chat_model(model) #Part2: Defino mi LLM

        self.llm_queue = Queue() #Part2: Para almacenar lo que respondió mi LLM

        self.create_agent()

    #==============================================================================
    #====================================  Paso 1  ================================
    def start_or_stop_recording(self):
        if self.is_recording: #Si estoy grabando
            self.is_recording = False #Quiero parar de grabar
            self.save_and_transcribe() #Vamos a guardar el audio y transcribir
            self.audio_data = [] #Luego elimino lo que grabé, porque quiero comenzar de nuevo a grabar tal vez
        else:
            print("Starting record") #Si no estoy grabando, EMPIEZO
            self.audio_data = [] #Listo para guardar lo que hable
            self.is_recording = True #Comienzo a grabar

    #==============================================================================
    #====================================  Paso 3  ================================
    #Ese texto transcrito lo enviamos para el Agente
    def create_agent(self):
        df = pd.read_csv("Datos/df_rent.csv")

        system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"El dataset tiene {len(df):,} filas y {len(df.columns)} columnas: "
            f"{', '.join(df.columns.tolist())}.\n"
            f"Primeras filas de muestra:\n{df.head(3).to_string(index=False)}"
        )

        @tool
        def query_dataframe(code: str) -> str:
            """Ejecuta código Python/pandas sobre el dataframe 'df' y retorna el resultado.
            El dataframe ya está disponible como variable 'df'. Asigna el resultado final
            a una variable llamada 'result'. Ejemplo: result = df.head()"""
            try:
                local_vars = {"df": df, "pd": pd}
                exec(code, {"pd": pd}, local_vars)
                result = local_vars.get("result", "Código ejecutado correctamente")
                return str(result)
            except Exception as e:
                return f"Error al ejecutar el código: {str(e)}"

        self.agent = create_agent(
            self.llm,
            tools=[query_dataframe],
            system_prompt=system_prompt,
        )
    
    #==============================================================================
    #===================================  Paso 2  =================================
    #Guardamos nuestro audio y transcribimos para texto
    def save_and_transcribe(self):
        print("Saving the recording...")
        if "temp.wav" in os.listdir(): os.remove("temp.wav") #Si tengo un archivo de audio temporal lo elimino
        wav_file = wave.open("test.wav", 'wb') #Ahora creo un archivo de audio
        wav_file.setnchannels(self.channels)
        wav_file.setsampwidth(2)  # Corregido para usar la longitud de muestra para int16 directamente
        wav_file.setframerate(self.samplerate)
        wav_file.writeframes(np.array(self.audio_data, dtype=self.dtype)) #Transfiere mi audio_data a un array
        wav_file.close()

        result = self.whisper.transcribe("test.wav", fp16=False)
        print("Usuario:", result["text"])

        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": result["text"]}]},
            config={"recursion_limit": 50},
        )

        ai_message = "Sin respuesta."
        for msg in reversed(response.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                ai_message = msg.content
                break

        print("AI:", ai_message)
        self.llm_queue.put(ai_message)

    #=================================  Paso 4  ==============================
    #El agente devuelve una respuesta que es pasada a esta función para que sea reproducida
    def convert_and_play(self):
        tts_text = ''
        while True:
            tts_text += self.llm_queue.get()

            if '.' in tts_text or '?' in tts_text or '!' in tts_text:
                print(tts_text)
                
                spoken_response = client.audio.speech.create(model="tts-1",
                voice='alloy', 
                response_format="opus",
                input=tts_text
                )

                buffer = io.BytesIO()
                for chunk in spoken_response.iter_bytes(chunk_size=4096):
                    buffer.write(chunk)
                buffer.seek(0)

                with sf.SoundFile(buffer, 'r') as sound_file:
                    data = sound_file.read(dtype='int16')
                    sd.play(data, sound_file.samplerate)
                    sd.wait()
                tts_text = ''



    #===================== FUNCION PRINCIPAL (ORQUESTADOR) ======================
    def run(self):
        print("Estoy corriendo")
        t1 = threading.Thread(target=self.convert_and_play)
        t1.start()

        #Esta parte de aquí es bien difícil de implementar si no fuera por la documentación que explica bastante.

        def callback(indata, frame_count, time_info, status): #Copiado de la documentación de sounddevice
            if self.is_recording: #Copiado de la documentación de sounddevice
                self.audio_data.extend(indata.copy()) #Copiado de la documentación de sounddevice

        #Abrimos una instancia de Grabación de Audio en formato de Stream
        with sd.InputStream(samplerate=self.samplerate, #Copiado de la documentación de sounddevice
                            channels=self.channels, #Copiado de la documentación de sounddevice
                            dtype=self.dtype , #Copiado de la documentación de sounddevice
                            callback=callback): #Copiado de la documentación de sounddevice
            
            def on_activate(): #Copiado de la documentación de pynput
                self.start_or_stop_recording() #EDITADO

            def for_canonical(f): #Copiado de la documentación de pynput
                return lambda k: f(l.canonical(k)) #Copiado de la documentación de pynput

            hotkey = keyboard.HotKey( #Copiado de la documentación de pynput
                keyboard.HotKey.parse('<cmd>'), #EDITADO
                on_activate) #Copiado de la documentación de pynput
            with keyboard.Listener( #Copiado de la documentación de pynput
                    on_press=for_canonical(hotkey.press), #Copiado de la documentación de pynput
                    on_release=for_canonical(hotkey.release)) as l: #Copiado de la documentación de pynput
                l.join() #Copiado de la documentación de pynput


if __name__ == "__main__":
    #Instanciamos nuestra clase con este objeto
    talking_llm = TalkingLLM()
    #El Objeto llama a la función run que heredó
    talking_llm.run()
