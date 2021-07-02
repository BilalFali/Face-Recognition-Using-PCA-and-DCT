# Generated by Django 3.1.5 on 2021-04-06 18:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Client', '0003_auto_20210406_1951'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='client',
            options={'verbose_name': 'Client', 'verbose_name_plural': 'Clients'},
        ),
        migrations.AlterField(
            model_name='client',
            name='first_name',
            field=models.CharField(max_length=100, verbose_name='Client First Name'),
        ),
        migrations.AlterField(
            model_name='client',
            name='last_name',
            field=models.CharField(max_length=100, verbose_name='Client Last Name'),
        ),
        migrations.AlterField(
            model_name='client',
            name='username',
            field=models.CharField(max_length=100, verbose_name='Client UserName'),
        ),
    ]