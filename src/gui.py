import PySimpleGUI as sg
import script 
sg.theme("DarkTeal2")
column_to_be_centered = [[sg.T("")], [sg.Text("Choose Number of Depth Propagations ")],
          [sg.T(""), sg.Radio('1', "RADIO1", default=False),sg.Radio('2', "RADIO1", default=False),sg.Radio('3', "RADIO1", default=False),
          sg.Radio('4', "RADIO1", default=False),sg.Radio('5', "RADIO1", default=False),sg.Radio('6', "RADIO1", default=False)],
          [sg.T("")], [sg.Text("Choose Filter Size")],
          [sg.T(""), sg.Radio('3', "RADIO2", default=False),sg.Radio('5', "RADIO2", default=False),sg.Radio('7', "RADIO2", default=False),
          sg.Radio('9', "RADIO2", default=False),sg.Radio('11', "RADIO2", default=False),sg.Radio('13', "RADIO2", default=False)],
          [sg.T("")], [sg.Text("Write Value for SigmaS: "), sg.Input(key="sigmaS" ,change_submits=True)],
          [sg.T("")], [sg.Text("Write Value for SigmaR: "), sg.Input(key="sigmaR" ,change_submits=True)],
          [sg.T("")], [sg.Text("Choose RGB Image: "), sg.Input(), sg.FileBrowse(key="rgb")],
          [sg.T("")], [sg.Text("Choose Ground Truth Depth Image: "), sg.Input(), sg.FileBrowse(key="ground_truth")],[sg.Button("Submit")]]

layout = [[sg.Text(key='-EXPAND-', font='ANY 1', pad=(0, 0))],  # the thing that expands from top
              [sg.Text('', pad=(0,0),key='-EXPAND2-'),              # the thing that expands from left
               sg.Column(column_to_be_centered, vertical_alignment='center', justification='center',  k='-C-')]]

window = sg.Window('Bilateral Upscaler', layout, resizable=True,finalize=True)
window['-C-'].expand(True, True, True)
window['-EXPAND-'].expand(True, True, True)
window['-EXPAND2-'].expand(True, False, True)


###Setting Window
# window = sg.Window('Upscaling Parameters', layout, size=(1500,1500))

###Showing the Application, also GUI functions can be placed here.

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def popup(message, title):
    print("Please ensure all parameters are entered correctly")
    # sg.Window("Other Window", [[sg.Text("Try Again?")], [sg.Yes(), sg.No()]])
    smol_col = [[sg.Text(message)], [sg.OK()]]
    smol_layout = layout = [[sg.Text(key='-EXPANDS-', font='ANY 1', pad=(0, 0))],  # the thing that expands from top
        [sg.Text('', pad=(0,0),key='-EXPANDS2-'),              # the thing that expands from left
        sg.Column(smol_col, vertical_alignment='center', justification='center',  k='-CS-')]]
    smol_window = sg.Window(title, layout, resizable=True,finalize=True)
    smol_window['-CS-'].expand(True, True, True)
    smol_window['-EXPANDS-'].expand(True, True, True)
    smol_window['-EXPANDS2-'].expand(True, False, True)

    # if sg.Window("Other Window", smol_layout, size=(300,300), element_justification='c').read(close=True)[0] == "Yes":
    if smol_window.read(close=True)[0] == "Yes":
        print("User chose yes!")

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        break
    elif event=="Submit":
        print(values)
        scale=-1
        k=-1
        for i in range(0,6):
            if values[i]==True:
                scale=i+1
                break
        temp=3
        for i in range(6,12):
            if values[i]==True:
                k=temp
                break
            temp+=2
        if scale==-1:
            popup("Please choose an upscaling factor", 'Invalid Parameters')
            continue
        elif k==-1:
            popup("Please choose a filter size", 'Invalid Parameters')
            continue
        elif not(is_number(values["sigmaS"])):
            popup("Please enter a floating point value for Sigma S", 'Invalid Parameters')
            continue
        elif float(values['sigmaS']) <= 0:
            popup("Please ensure that Sigma S is greater than 0", 'Invalid Parameters')
            continue
        elif not(is_number(values["sigmaR"])):
            popup("Please enter a floating point value for Sigma R", 'Invalid Parameters')
            continue
        elif float(values['sigmaR']) <= 0:
            popup("Please ensure that Sigma R is greater than 0", 'Invalid Parameters')
            continue
        elif not values["rgb"] or not values["ground_truth"]:
            popup("Please select paths for input RGB and Depth images", 'Invalid Parameters')
            continue
        
        popup("Your image is ready to be processed, press OK to begin", 'Confirmation')
        # print("window done")
        #     continue
        # print(scale,k)
        script.centre(scale,k,float(values["sigmaS"]),float(values["sigmaR"]),values["rgb"],values["ground_truth"])
        # print("Hello World")
        # print(values["-IN99-"])
    
window.close()