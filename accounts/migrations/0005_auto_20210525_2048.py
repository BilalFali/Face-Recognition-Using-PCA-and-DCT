# Generated by Django 3.1.5 on 2021-05-25 19:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0004_auto_20210525_2046'),
    ]

    operations = [
        migrations.AlterField(
            model_name='parameter',
            name='minNeighbors',
            field=models.FloatField(default=4),
        ),
        migrations.AlterField(
            model_name='parameter',
            name='scaleFactor',
            field=models.FloatField(default=1.1),
        ),
    ]
