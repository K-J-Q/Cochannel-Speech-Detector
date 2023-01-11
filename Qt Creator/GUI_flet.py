import flet as ft
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart
import time
import random
import torch



def main(page: ft.Page):
    plotHistory = torch.zeros([10])
    spechHistory = torch.zeros([123,32*10])

    def runModel():
        spec = torch.rand([123, 32])
        pred = random.randint(0, 2)

        spechHistory = torch.cat((spechHistory[:,32:], spec), dim=1)
        plotHistory = torch.cat((plotHistory[1:], torch.tensor([pred])))

        ax1.imshow(specHistory)
        time.sleep(1)

        

    def startModel(e):
        if startStopButton.text == "Stop Model":
            startStopButton.text = "Start Model"
        else:
            startStopButton.text = "Stop Model"
        page.update()

        while startStopButton.text == "Stop Model":
            runModel()
            page.update()

    page.title = "Cochannel Speech Detector"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.imshow(specHistory)
    ax2.plot([1, 2, 3, 4], [1, 4, 9, 16])

    modelPred = ft.Text("0", size=120)
    startStopButton = ft.ElevatedButton(text="Start Model", on_click=startModel)

    page.add(
        ft.Row([
        ft.Column(
            [
                ft.Row([
                    ft.TextField(label="Audio Input"),
                    ft.IconButton(ft.icons.REFRESH),
                ]),
                modelPred,
                ft.Row([
                    ft.Text("0.05"),
                    ft.Text("0.15"),
                    ft.Text("0.8")
                ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND
                ),
                ft.Row([
                    ft.Text("Class 0"),
                    ft.Text("Class 1"),
                    ft.Text("Class 2")
                ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND
                ),
                startStopButton
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        ),
            MatplotlibChart(fig, expand=True)
        ])
    )

ft.app(target=main)