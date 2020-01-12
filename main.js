const electron = require('electron')
const log = require('electron-log');
const path = require('path')
const url = require('url')

const { app, BrowserWindow } = electron

/* python FLASK server in child process
 *
 *
 ** * * * * * * */

const PY_DIR = 'server'
const PY_APP = 'app'

let py_proc = null

let py_host = '0.0.0.0'
let py_port = 8081

global.PY_SERVER = {
    host: py_host,
    port: py_port
};

const get_script_path = () => {
    return path.join(__dirname, PY_DIR, PY_APP + '.py')
}

const create_py_proc = () => {
    let script = get_script_path()
    let host_arg = '--host=' + py_host
    let port_arg = '--port=' + py_port

    py_proc = require('child_process').spawn(
        'flask', ['run', host_arg, port_arg], { cwd: PY_DIR });
}


const exit_py_proc = () => {
    py_proc.kill()
    py_proc = null
    py_port = null
}

app.on('ready', create_py_proc)
app.on('will-quit', exit_py_proc)

/* electron js main process
 *
 *
 * * * * * * * */

const index_path = path.join(app.getAppPath(), 'src/index.html');

let window = null

app.once('ready', () => {
    const { width, height } = electron.screen.getPrimaryDisplay().workAreaSize
    window = new BrowserWindow({
        width: width,
        height: height,
        backgroundColor: '#D6D8DC',
        show: false,
        webPreferences: {
            nodeIntegration: true
        }
    })

    window.webContents.openDevTools()

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