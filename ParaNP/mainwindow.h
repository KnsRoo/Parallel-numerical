#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <stdint.h>
#include <stdio.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void OnSlider();
    void OnSetClicked();
    void OnSliderReleased();
    void OnComboChanged();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
