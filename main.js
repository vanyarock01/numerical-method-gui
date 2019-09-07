const electron = require('electron')
const log = require('electron-log');
const path = require('path')
const url = require('url')

const { app, BrowserWindow } = electron
const index_path = path.join(app.getAppPath(), 'src/index.html');

let window = null

app.once('ready', () => {
    const { width, height } = electron.screen.getPrimaryDisplay().workAreaSize
    window = new BrowserWindow({
        width: width,
        height: height,
        backgroundColor: "#D6D8DC",
        show: false
    })
    
    window.setMenuBarVisibility(false)

    window.loadURL(url.format({
        pathname: index_path,
        protocol: 'file:',
        slashes: true
    }))

    window.once('ready-to-show', () => {
        window.show()
    })
})