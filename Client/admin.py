from django.contrib import admin
from .models import Client
from Client.models import Dataset,Training

# Register your models here.
#class UserDataSetAdmin(admin.StackedInline):
    #model = DataSetUser

#@admin.register(User)
#class PostAdmin(admin.ModelAdmin):
   # inlines = [UserDataSetAdmin]

    #class Meta:
      # model = User

#@admin.register(DataSetUser)
#class PostImageAdmin(admin.ModelAdmin):
  #  pass

admin.site.register(Client)
admin.site.register(Dataset)
admin.site.register(Training)

