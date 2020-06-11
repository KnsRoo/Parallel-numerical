#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "library.h"
#include <iostream>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->set, SIGNAL(clicked()), this, SLOT(OnSetClicked()));
    connect(ui->f, SIGNAL(clicked()), this, SLOT(OnFSClicked()));
    connect(ui->k, SIGNAL(clicked()), this, SLOT(OnFSClicked()));
    connect(ui->g, SIGNAL(clicked()), this, SLOT(OnTClicked()));
    connect(ui->slider, SIGNAL(sliderReleased()), this, SLOT(OnSliderReleased()));
    connect(ui->slider, SIGNAL(valueChanged(int)), this, SLOT(OnSliderReleased()));
    connect(ui->method, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboChanged()));
}

void MainWindow::OnFSClicked(){
    if (ui->method->currentIndex() == 5){
        ui->slider->setMaximum(10000000);
    }
    else
        ui->slider->setMaximum(1000000);
}

void MainWindow::OnTClicked(){
    if (ui->method->currentIndex() == 5){
        ui->slider->setMaximum(10000000);
    }
    else
        ui->slider->setMaximum(100000);
}


void MainWindow::OnComboChanged(){
    if (ui->g->isChecked()){
        if (ui->method->currentIndex() == 5){
            ui->slider->setMaximum(10000000);
        } else{
            ui->slider->setMaximum(100000);
        }
    } else {
        if (ui->method->currentIndex() == 5){
            ui->slider->setMaximum(10000000);
        } else{
            ui->slider->setMaximum(1000000);
        }
    }
}

void MainWindow::OnSliderReleased(){
    if (ui->slider->value() % 2 != 0) ui->slider->setValue(ui->slider->value()+1);
     ui->parts->setText("Количество разбиений: " + QString::number(ui->slider->value()));
}

void MainWindow::OnSlider(){
     ui->parts->setText("Количество разбиений: " + QString::number(ui->slider->value()));
}

void MainWindow::OnSetClicked(){
    int index = ui->method->currentIndex();
    int parts = ui->parts->text().remove(0,22).toInt();
    double result = 0.0, innac = 0.0;
    if (ui->f->isChecked()){
        result = calculate(index,parts,true);
        innac = inaccuracy(index,parts,true);
    }
    if (ui->k->isChecked()){
        result = calculate(index,parts,false);
        innac = inaccuracy(index,parts,false);
    }
    if (ui->g->isChecked()){
        result = calculate_xy(index,parts);
        innac = inaccuracy_xy(index,parts);
    }
    ui->result->setText(QString::number(result));
    ui->gamma->setText(QString::number(innac));
}

MainWindow::~MainWindow()
{
    delete ui;
}
