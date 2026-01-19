**Multimodal VQA (Visual Question Answering) AI Agent**
---

## Proje Amacı
Bu projede, metin tabanlı çalışan **Llama 3.1** modeline görsel algı yeteneği kazandırmak için, **BLIP** ve **CLIP** modellerini sisteme entegre ettim

## Mimari
1.  **BLIP**
    * Resme bakar ve ne gördüğünü yazıya döker.
    * *Rolü:* Sistemin "Gözü".
2.  **CLIP**
    * BLIP'in gördüğü şey ile resim arasındaki anlamsal bağı kontrol eder.
    * Güven skoru üretir.
    * *Rolü:* Sistemin "Kontrolcüsü".
3.  **LLaMA 3.1** 
    * BLIP'in ürettiği görsel betimlemeyi ve kullanıcı sorusunu sentezleyerek cevap üretir.
    * *Rolü:* Sistemin "Beyni".

---
##  Kullanılan Teknolojiler ve Modeller

| Bileşen | Teknoloji / Model | Link|
| :--- | :--- | :--- |
| **Backend** | FastAPI | |
| **Frontend** | Streamlit | |
| **LLM** | **Llama 3.1 (8B)** | [Llama](https://ollama.com/library/llama3.1) |
| **BLIP** | `Salesforce/blip-image-captioning-large` | [Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-large) |
| **CLIP** | `openai/clip-vit-base-patch32` | [Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32) |

---

##  Kurulum 

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### 1. Gereksinimler
* Python 3.10 veya üzeri
* [Ollama](https://ollama.com/) (Llama modelini çalıştırmak için)
* Gerekli kütüphaneleri yükleyin.
```bash
  pip install -r requirements.txt
   ```
### 2. env Dosyası
* OLLAMA_URL = Python kodunun, yerel bilgisayarınızda çalışan Ollama ile konuşmasını sağlar.
* LOCAL_URL = Streamlit arayüzünün, resim ve soruları gönderdiği FastAPI sunucu adresidir.
* LLAMA_MODEL = Llama modelinizin ismini yazın.
* CLIP_MODEL = CLIP modelinizin ismini yazın.
* BLIP_MODEL = BLIP modelinizin ismini yazın.

### 3. Çalıştırma
İki ayrı terminal açıp sırasıyla terminallere aşağıdakileri yazın.
* uvicorn app.main:app --reload
* streamlit run ui.py

## ⚠️ Uyarı
## Bu proje hiçbir şekilde ticari amaç içermemektedir.
