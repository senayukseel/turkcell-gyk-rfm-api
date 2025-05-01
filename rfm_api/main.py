from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import io 
import base64
import matplotlib.pyplot as plt

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

kmeans_model = joblib.load("kmeans.pkl")

cluster_labels = kmeans_model.labels_ 
cluster_counts = [sum(cluster_labels == i) for i in range(kmeans_model.n_clusters)]

cluster_profiles = {
    0: "Bu küme, düşük harcama yapan ve seyrek alışveriş gerçekleştiren müşterilerden oluşmaktadır. - Pasif veya kaybedilmiş müşteri profili",
    1: "Bu küme, yüksek harcama yapan ve sık alışveriş gerçekleştiren sadık müşterilerden oluşmaktadır. - En iyi müşteriler",
    2: "Bu küme, son zamanlarda ara sıra alışveriş yapan, potansiyel olarak sadık müşteri olabilecek kişilerden oluşmaktadır.",
    3: "Bu küme, geçmişte belirli bir sıklıkta alışveriş yapmış ancak uzun süredir alışveriş yapmamış müşterilerden oluşmaktadı. - Geri kazanılabilir eski müşteriler",
    4: "Bu küme, son dönemde çok sık alışveriş yapmış ve yüksek sadakat göstermiş aktif müşterilerden oluşmaktadır. - VIP müşteriler "
}


def pie_chart(prediction):
    cluster_counts = [sum(kmeans_model.labels_ == i) for i in range(kmeans_model.n_clusters)]
    cluster_labels = [f"{i}. Grup" for i in range(len(cluster_counts))]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('equal')

    wedges, texts, autotexts = ax.pie(
        cluster_counts, 
        labels=None, 
        autopct='%1.1f%%', 
        startangle=90,
        pctdistance=0.77 
    )

    predicted_cluster = prediction[0] 
    wedges[predicted_cluster].set_edgecolor('black')  
    wedges[predicted_cluster].set_facecolor('yellow')  

    for i, autotext in enumerate(autotexts):
        autotext.set_fontsize(15)  
        autotext.set_color('black') 
        autotext.set_weight('bold' if i == predicted_cluster else 'normal') 

        x, y = autotext.get_position()
        
        if i % 2 == 0:
            autotext.set_position((x, y + 0.05)) 
        else:
            autotext.set_position((x, y - 0.05))


    ax.legend(
        handles=[
            plt.Line2D(
                [0], [0], 
                marker='o', color='w',
                markerfacecolor=(wedges[i].get_facecolor()),
                markersize=16,
                markeredgewidth=2,
                markeredgecolor='black' if i == predicted_cluster else 'white'
            )
            for i in range(len(wedges))
        ],
        labels=cluster_labels,
        title="Kümeler",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=12,
        title_fontsize=14,
        frameon=True
    )

    plt.tight_layout(pad=2)
    
    # Görseli base64 formatında döndür
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_base64 = base64.b64encode(img_stream.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{img_base64}"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Başlangıçta herhangi bir tahmin ve veri yok, form alanları boş olacak
    return templates.TemplateResponse("form.html", {
        "request": request, "recency": None, "frequency": None, "monetary": None, "cluster": None, "error": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, recency: float = Form(...), frequency: float = Form(...), monetary: float = Form(...)):
    try: 
        if recency < 0 or frequency < 0 or monetary < 0:
            raise ValueError("Tüm RFM değerleri sıfır veya daha büyük olmalıdır.")

        data = np.array([[recency, frequency, monetary]]) 
        prediction = kmeans_model.predict(data)
        
        cluster_profile = cluster_profiles.get(int(prediction[0]), "Bu küme hakkında bilgi mevcut değil.")
        graph_url = pie_chart(prediction)
        return templates.TemplateResponse("form.html", {
            "request": request,
            "cluster": int(prediction[0]),
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "error": None,
            "cluster_profile": cluster_profile,  
            "graph_url": graph_url  
        })
    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "cluster": None,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "error": str(e)
        })  
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
#Python dosyasını çalıştırdığımızda uvicorn server'ı başlatmak ve FastAPI uygulamasını doğru bir şekilde çalıştırmak için kullanıldı.
