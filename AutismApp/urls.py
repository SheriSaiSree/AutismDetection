from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	
	       path('LoadDataset', views.LoadDataset, name="LoadDataset"),
	       path('LoadDatasetAction', views.LoadDatasetAction, name="LoadDatasetAction"),	
	       path('TrainModel', views.TrainModel, name="TrainModel"),
	       path('DetectAutism', views.DetectAutism, name="DetectAutism"),
	       path('DetectAutismAction', views.DetectAutismAction, name="DetectAutismAction"),	       
]