from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import *
from src.visualization.insights import generate_all_insights 
from src.visualization.interactive_insights import generate_all_html_charts 
from src.visualization.interactive_elements import load_data, filter_data, generate_chart 
from src.storytelling.analysis_text import generate_story_insights

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()


templates = Jinja2Templates(directory='templates')


origins = ["*"]

app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.text: Optional[str] = None

    async def get_text_data(self):
        form =  await self.request.form()
        self.text = form.get('input_text')
        

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/")
async def predictGetRouteClient(request: Request):
    try:

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": "Rendering"},
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
@app.get("/predict")
async def predictGetRouteClient(request: Request):
    try:

        return templates.TemplateResponse(
            "prediction.html",
            {"request": request, "context": False},
        )
        
    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        
        await form.get_text_data()
        
        input_data = [form.text]
        print(form.text)
        
        # return Response(f"got data is : {input_data[0]}")
    
        
        prediction_pipeline = PredictionPipeline()
        prediction: int = prediction_pipeline.run_pipeline(input_data=input_data)
        
        print(f"the prediction is : {prediction}")
       
        
        return templates.TemplateResponse(
            "prediction.html",
            {"request": request, "context": True, "prediction": prediction[0]}
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}
    
    @app.get("/dashboard")
    async def dashboard_route(request: Request):
        try:
            charts = generate_all_insights()
            return templates.TemplateResponse("dashboard.html", {"request": request, **charts})
        except Exception as e:
            return Response(f"Error generating dashboard: {e}")
        
# Mount static files directory for serving static content
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/dashboard")
async def dashboard_route(request: Request):
    charts = generate_all_html_charts()
    return templates.TemplateResponse("dashboard.html", {"request": request, **charts})

@app.get("/interactive")
async def interactive_get(request: Request):
    df = load_data()
    filtered_df = filter_data(df, "all")
    chart_html = generate_chart(filtered_df, "bar")
    return templates.TemplateResponse("interactive.html", {
        "request": request,
        "chart": chart_html,
        "selected_filter": "all",
        "selected_chart": "bar"
    })

@app.post("/interactive")
async def interactive_post(
    request: Request,
    label_filter: str = Form(...),
    chart_type: str = Form(...)
):
    df = load_data()
    filtered_df = filter_data(df, label_filter)
    chart_html = generate_chart(filtered_df, chart_type)

    return templates.TemplateResponse("interactive.html", {
        "request": request,
        "chart": chart_html,
        "selected_filter": label_filter,
        "selected_chart": chart_type
    })

@app.get("/story-dashboard")
async def story_dashboard(request: Request):
    insights = generate_story_insights()
    return templates.TemplateResponse("story_dashboard.html", {
        "request": request,
        "insights": insights
    })

if __name__ == "__main__":
    app_run(app, host = APP_HOST, port =APP_PORT)
    
