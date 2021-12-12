from tkinter import *

class App(object):
    def __init__(self, **kwargs):
        self.winx, self.winy = 0, 0
        self.width, self.height = kwargs.get("width", 400), kwargs.get("height", 400)
        self.timerDelay = 100
        self.dragDelay = 50
        self.running = False
        self.activeMode = Mode()
        self.run()

    def bindRoot(self):
        self.root = Tk()
        self.root.resizable(False, False)
        self.root.createcommand('exit', lambda: '')
        self.root.protocol('WM_DELETE_WINDOW', lambda: self.quit())
        self.root.bind("<Button-1>", lambda event: self.mousePressedWrapper(event))
        self.root.bind("<B1-ButtonRelease>", lambda event: self.mouseReleasedWrapper(event))
        self.root.bind("<KeyPress>", lambda event: self.keyPressedWrapper(event))
        self.root.app = self

    def setupCanvas(self):
        self.root.geometry(f'{self.width}x{self.height}+{self.winx}+{self.winy}')
        self.root.canvas = Canvas()
        self.root.canvas.pack(fill=BOTH, expand=YES)

    def setupVariables(self):
        self.mouseIsPressed = False
        self.lastMousePosn = (-1, -1)
        self.afterIdMap = dict()
        self.running = True

    def kickstart(self):
        self.appStarted()
        self.timerFiredWrapper()
        self.mouseMotionWrapper()
        self.root.update()
        self.root.deiconify()
        self.root.lift()
        self.root.focus()
        self.root.mainloop()

    def quit(self):
        self.root.quit()
        self.root.withdraw()
        self.running = False
        for afterId in self.afterIdMap:
            self.root.after_cancel(self.afterIdMap[afterId])
        self.afterIdMap.clear()
        self.activeMode.appStopped()

    def run(self):
        self.bindRoot()
        self.setupCanvas()
        self.setupVariables()
        self.kickstart()

    def deferredMethodCall(self, afterId, afterDelay, afterFn):
        def afterFnWrapper():
            self.afterIdMap.pop(afterId, None)
            afterFn()
        id = self.afterIdMap.get(afterId, None)
        if ((id is None)):
            if id: self.root.after_cancel(id)
            self.afterIdMap[afterId] = self.root.after(afterDelay, afterFnWrapper)

    def timerFiredWrapper(self):
        if not self.running: return
        if self.activeMode.hasOverwritten("timerFired"):
            self.activeMode.timerFired()
            self.redrawAllWrapper()
        afterId = "timerFiredWrapper"
        def afterFnWrapper():
            self.afterIdMap.pop(afterId, None)
            self.timerFiredWrapper()
        if self.afterIdMap.get(afterId, None) == None:
            self.afterIdMap[afterId] = self.root.after(self.timerDelay, afterFnWrapper)

    def mousePressedWrapper(self, event):
        if not self.running: return
        if ((event.x >= 0) and (event.x <= self.width) and
            (event.y >= 0) and (event.y <= self.height)):
            self.mouseIsPressed = True
            if self.activeMode.hasOverwritten("mousePressed"):
                self.activeMode.mousePressed(event)
                self.redrawAllWrapper()

    def mouseReleasedWrapper(self, event):
        if not self.running: return
        if ((event.x >= 0) and (event.x <= self.width) and
            (event.y >= 0) and (event.y <= self.height)):
            self.mouseIsPressed = False
            if self.activeMode.hasOverwritten("mouseReleased"):
                self.activeMode.mouseReleased(event)
                self.redrawAllWrapper()

    def mouseMotionWrapper(self):
        if not self.running: return
        oldX, oldY = self.lastMousePosn
        newX = self.root.winfo_pointerx() - self.root.winfo_rootx()
        newY = self.root.winfo_pointery() - self.root.winfo_rooty()
        if ((newX >= 0) and (newX <= self.width) and
            (newY >= 0) and (newY <= self.height) and
            (newX, newY) != (oldX, oldY)):
            event = type("MouseEvent", (object,), {"x": newX, "y": newY})
            self.lastMousePosn = (newX, newY)
            if self.mouseIsPressed and self.activeMode.hasOverwritten("mouseDragged"):
                self.activeMode.mouseDragged(event)
                self.redrawAllWrapper()
            if not self.mouseIsPressed and self.activeMode.hasOverwritten("mouseMoved"):
                self.activeMode.mouseMoved(event)
                self.redrawAllWrapper()
        afterId = "mouseMotionWrapper"
        def afterFnWrapper():
            self.afterIdMap.pop(afterId, None)
            self.mouseMotionWrapper()
        if self.afterIdMap.get(afterId, None) == None:
            self.afterIdMap[afterId] = self.root.after(self.dragDelay, afterFnWrapper)

    class KeyEventWrapper(Event):
        def __init__(self, event):
            keyNameMap = { '\t':'Tab', '\n':'Enter', '\r':'Enter', '\b':'Backspace',
                   chr(127):'Delete', chr(27):'Escape', ' ':'Space' }
            keysym, char = event.keysym, event.char
            del event.keysym
            del event.char
            for key in event.__dict__:
                if (not key.startswith('__')):
                    self.__dict__[key] = event.__dict__[key]
            key = c = char
            hasControlKey = (event.state & 0x4 != 0)
            if ((c in [None, '']) or (len(c) > 1) or (ord(c) > 255)):
                key = keysym
                if (key.endswith('_L') or
                    key.endswith('_R') or
                    key.endswith('_Lock')):
                    key = 'Modifier_Key'
            elif (c in keyNameMap):
                key = keyNameMap[c]
            elif ((len(c) == 1) and (1 <= ord(c) <= 26)):
                key = chr(ord('a')-1 + ord(c))
                hasControlKey = True
            if hasControlKey and (len(key) == 1):
                key = 'control-' + key
            self.key = key
        keysym = property(lambda *args: App._useEventKey('keysym'),
                          lambda *args: App._useEventKey('keysym'))
        char =   property(lambda *args: App._useEventKey('char'),
                          lambda *args: App._useEventKey('char'))

    def keyPressedWrapper(self, event):
        if not self.running: return
        event = App.KeyEventWrapper(event)
        if event.key != 'Modifier_Key' and self.activeMode.hasOverwritten("keyPressed"):
            self.activeMode.keyPressed(event)
            self.redrawAllWrapper()

    def redrawAllWrapper(self):
        self.root.canvas.delete(ALL)
        self.activeMode.redrawAll(self.root.canvas)
        self.root.canvas.update()

    def setMode(self, mode):
        mode = self.__dict__[mode]
        self.activeMode = mode
        if not mode.initialized:
            mode.width, mode.height, mode.root = self.width, self.height, self.root
            mode.parent = self
            mode.appStarted()
            self.redrawAllWrapper()
            mode.initialized = True

class Mode(object):
    def __init__(self):
        self.initialized = False

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except:
            return self.parent.__getattribute__(attr)

    def setMode(self, mode):
        self.parent.setMode(mode)

    def quit(self):
        self.parent.quit()

    def hasOverwritten(self, method):
        return (getattr(type(self), method) is not getattr(Mode, method))

    def appStarted(app): pass
    def appStopped(app): pass
    def timerFired(app): pass
    def mousePressed(app, event): pass
    def mouseReleased(app, event): pass
    def mouseDragged(app, event): pass
    def mouseMoved(app, event): pass
    def keyPressed(app, event): pass
    def redrawAll(app, canvas): pass

    def getMousePos(self):
        x = self.root.winfo_pointerx() - self.root.winfo_rootx()
        y = self.root.winfo_pointery() - self.root.winfo_rooty()
        return x, y

class Button(object):
    def __init__(self, x, y, w, h, text, color1, color2, onClick):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text, self.color1, self.color2 = text, color1, color2
        self.onClick = onClick
        self.colors = [self.color1, self.color2]

    def hover(self, x, y):
        if self.x - self.w / 2 < x < self.x + self.w / 2:
            if self.y - self.h / 2 < y < self.y + self.h / 2:
                self.colors = [self.color2, self.color1]
                return True
        self.colors = [self.color1, self.color2]
        return False

    def click(self, x, y):
        if self.x - self.w / 2 < x < self.x + self.w / 2:
            if self.y - self.h / 2 < y < self.y + self.h / 2:
                self.onClick()
                return True
        return False

    def render(self, canvas, **kwargs):
        hW = kwargs.get("highlightWidth", 0)
        hC = kwargs.get("highlightColor", "")
        font = kwargs.get("font", "Futura 12 bold")
        shape = kwargs.get("shape", "rectangle")
        if shape == "rectangle":
            drawFn = canvas.create_rectangle
        elif shape == "oval":
            drawFn = canvas.create_oval
        drawFn(self.x - self.w / 2, self.y - self.h / 2, 
        self.x + self.w / 2, self.y + self.h / 2, fill=self.colors[0], width=0)
        drawFn(self.x - self.w / 2 + hW/2, self.y - self.h / 2 + hW/2, 
        self.x + self.w / 2 - hW/2, self.y + self.h / 2 - hW/2, outline=hC, width=hW)
        canvas.create_text(self.x, self.y, text=self.text, font=font,
        fill=self.colors[1])

class ScrollBar(object):
    def __init__(self, x0, y0, x1, y1, alias):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.alias = alias
        self.holding = False
        self.w = self.x1 - self.x0

    def click(self, x, y):
        self.holding = self.x0 < x < self.x1 and self.y0 < y < self.y1

    def release(self):
        self.holding = False

    def scroll(self, x, y):
        if self.holding:
            y = min(max(self.y0 + self.w/2, y), self.y1 - self.w/2)
            self.alias[0] = (y - (self.y0 + self.w/2)) / (self.y1 - self.y0 - self.w)

    def render(self, canvas, **kwargs):
        background = kwargs.get("background", "#eeeeee")
        foreground = kwargs.get("foreground", "#aaaaaa")
        buffer = kwargs.get("buffer", 3)
        canvas.create_rectangle(self.x0, self.y0, self.x1, self.y1, 
        fill=background, width=0)
        r = (self.x1 - self.x0)/2 - buffer
        y = self.y0 + self.w/2 + (self.y1 - self.y0 - self.w) * self.alias[0]
        canvas.create_oval(self.x0 + buffer, y-r, self.x1-buffer, y+r, 
        fill=foreground, width=0)
