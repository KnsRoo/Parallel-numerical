#include "mainwindow.h"
#include <QApplication>
#include <QFontDatabase>
#include <math.h>
#include <stdint.h>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QFontDatabase::addApplicationFont("interface/HelveticaNeueCyr-Roman.ttf");
    MainWindow w;
    w.show();

    return a.exec();
}
