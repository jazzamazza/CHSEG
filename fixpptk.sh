echo "Move file"
mv ./venv/lib/python3.6/site-packages/pptk/libs/qt_plugins/platforms/libqxcb.so ./venv/lib/python3.6/site-packages/pptk/libs/qt_plugins/platforms/libqxcb.so.old
echo "Link file"
ln -s /usr/lib/qt/plugins/platforms/libqxcb.so ./venv/lib/python3.6/site-packages/pptk/libs/qt_plugins/platforms/libqxcb.so
echo "Done"
echo "Move file"
mv ./venv/lib/python3.6/site-packages/pptk/libs/libz.so.1 ./venv/lib/python3.6/site-packages/pptk/libs/libz.so.1.old
echo "Link file"
ln -s /usr/lib/libz.so.1 ./venv/lib/python3.6/site-packages/pptk/libs/libz.so.1
echo "Done"
