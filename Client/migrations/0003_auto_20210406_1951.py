# Generated by Django 3.1.5 on 2021-04-06 18:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Client', '0002_auto_20210404_2155'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='User',
            new_name='Client',
        ),
        migrations.DeleteModel(
            name='Admin',
        ),
    ]