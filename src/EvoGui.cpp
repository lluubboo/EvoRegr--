#include <FL/Fl.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Radio_Round_Button.H>
#include <FL/Fl_Multiline_Output.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Pack.H>
#include <FL/Fl_Terminal.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Text_Buffer.H>
#include <FL/Fl_File_Chooser.H>
#include <future>
#include <string>
#include "EvoGui.hpp"
#include "Log.hpp"

//*************************************************************************************************constants

static constexpr const char* WINDOW_NAME = "EVOREGR++ 1.0.0";
static constexpr int WINDOW_WIDTH = 1500;
static constexpr int WINDOW_HEIGHT = 600;
static constexpr int MAIN_WIDGET_PACK_WIDTH = 200;
static constexpr int BUTTON_HEIGHT = 30;
static constexpr int SPACER_HEIGHT = 0;
static constexpr int LABEL_HEIGHT = 25;
static constexpr int MENU_HEIGHT = 30;

/**
 * @brief Constructs a new EvoView object.
 *
 * This constructor initializes a new EvoView object, which represents the main window of the application.
 * It sets the appearance of the window, renders the main window, and initializes the logger.
 *
 * @param width The width of the window in pixels.
 * @param height The height of the window in pixels.
 * @param title The title of the window.
 */
EvoView::EvoView(int width, int height, const char* title) :
    Fl_Window(width, height, title),
    decomposition_method{ "LDLT" },
    generations_count{ 100 },
    generations_size{ 100 },
    interference_size{ 0 },
    mutation_rate{ 15 },
    island_count{ 12 },
    migration_ratio{ 5 },
    migration_interval{ 5 }
{
    set_appearance();
    render_main_window();
    //connect terminal to global project logger
    connect_terminal_to_logger();
}

//*************************************************************************************************callbacks

/**
 * @brief Handles the quit action.
 *
 * This function is called when the user triggers the quit action (e.g., by clicking a quit button).
 * It immediately terminates the program with a status of 0.
 *
 * @param w The FLTK widget that triggered the callback. This parameter is not used.
 * @param data User-defined data. This parameter is not used.
 */
void EvoView::quit_callback(Fl_Widget* /*w*/, void* /*data*/) {
    exit(0);
}

/**
 * @brief Displays a help window.
 *
 * This function is called when the user triggers the help action (e.g., by clicking a help button).
 * It creates a new window with a text display that contains contact information for help.
 *
 * @param w The FLTK widget that triggered the callback. This parameter is not used.
 * @param data User-defined data. This parameter is not used.
 */
void EvoView::help_callback(Fl_Widget* /*w*/, void* /*data*/) {
    Fl_Window* window = new Fl_Window(400, 200, "Help");
    Fl_Text_Buffer* buff = new Fl_Text_Buffer();
    Fl_Text_Display* disp = new Fl_Text_Display(20, 20, 360, 160, "Help");
    disp->buffer(buff);
    window->show();
    buff->text("For help, please contact: lubomirbalazjob@gmail.com");
}

/**
 * @brief Handles the input for the generations count.
 *
 * This function is called when the user inputs a value for the generations count.
 * It checks if the input value is a valid integer. If it is, it updates the `generations_count` field
 * and logs a message. If it's not, it logs an error message.
 *
 * @param w The FLTK widget (an Fl_Input instance) that triggered the callback.
 * @param v A pointer to the EvoView instance.
 */
void EvoView::gen_count_input_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    std::string input_value = ((Fl_Input*)w)->value();
    if (!input_value.empty() && std::all_of(input_value.begin(), input_value.end(), ::isdigit)) {
        T->generations_count = std::stoi(input_value);
        EvoRegression::Log::get_logger()->info("Generations count set to: " + std::to_string(T->generations_count));
    }
    else {
        EvoRegression::Log::get_logger()->error("Invalid input. Please enter a valid integer.");
    }
}

/**
 * @brief Handles the input for the generations size.
 *
 * This function is called when the user inputs a value for the generations size.
 * It checks if the input value is a valid integer. If it is, it updates the `generations_size` field
 * and logs a message. If it's not, it logs an error message.
 *
 * @param w The FLTK widget (an Fl_Input instance) that triggered the callback.
 * @param v A pointer to the EvoView instance.
 */
void EvoView::gen_size_input_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    std::string input_value = ((Fl_Input*)w)->value();
    if (!input_value.empty() && std::all_of(input_value.begin(), input_value.end(), ::isdigit)) {
        T->generations_size = std::stoi(input_value);
        EvoRegression::Log::get_logger()->info("Generations size set to: " + std::to_string(T->generations_size));
    }
    else {
        EvoRegression::Log::get_logger()->error("Invalid input. Please enter a valid integer.");
    }
}

/**
 * @brief Handles the input for the generations size.
 *
 * This function is called when the user inputs a value for the generations size.
 * It checks if the input value is a valid integer. If it is, it updates the `generations_size` field
 * and logs a message. If it's not, it logs an error message.
 *
 * @param w The FLTK widget (an Fl_Input instance) that triggered the callback.
 * @param v A pointer to the EvoView instance.
 */
void EvoView::gen_interference_size_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    std::string input_value = ((Fl_Input*)w)->value();
    if (!input_value.empty() && std::all_of(input_value.begin(), input_value.end(), ::isdigit)) {
        T->interference_size = std::stoi(input_value);
        EvoRegression::Log::get_logger()->info("Interference columns size set to: " + std::to_string(T->interference_size));
    }
    else {
        EvoRegression::Log::get_logger()->error("Invalid input. Please enter a valid integer.");
    }
}

/**
 * @brief Callback function for mutation rate input.
 *
 * This function is called when the mutation rate input box is modified. It validates the input to ensure it's a decimal number between 0 and 1. If the input is valid, it updates the mutation_rate member of the EvoView object and logs a message. If the input is not valid, it logs an error message.
 *
 * @param w The Fl_Widget that triggered the callback (the mutation rate input box).
 * @param v A void pointer to the EvoView object.
 */
void EvoView::mutation_rate_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    std::istringstream iss(((Fl_Input*)w)->value());
    int value;
    if (!(iss >> value) || !iss.eof() || value < 0 || value > 100) {
        EvoRegression::Log::get_logger()->error("Invalid input. Please enter an integer number between 0 and 100.");
    }
    else {
        T->mutation_rate = value;
        EvoRegression::Log::get_logger()->info("Mutation rate set to: " + std::to_string(T->mutation_rate));
    }
}

/**
 * Callback function for handling the island count input.
 * Parses the input value as an integer and sets the island count accordingly.
 * If the input is invalid (not an integer or negative), an error message is logged.
 * @param w The Fl_Widget object representing the input field.
 * @param v A pointer to the EvoView object.
 */
void EvoView::island_count_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    std::istringstream iss(((Fl_Input*)w)->value());
    int value;
    if (!(iss >> value) || !iss.eof() || value < 0) {
        EvoRegression::Log::get_logger()->error("Invalid input. Please enter an integer number.");
    }
    else {
        T->island_count = value;
        EvoRegression::Log::get_logger()->info("Island count set to: " + std::to_string(T->island_count));
    }
}

/**
 * Callback function for handling the migration ratio input.
 * Parses the input value as an integer and sets it as the migration ratio.
 * If the input is invalid, an error message is logged.
 * @param w The Fl_Widget triggering the callback.
 * @param v The user data associated with the widget.
 */
void EvoView::migration_ratio_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    std::istringstream iss(((Fl_Input*)w)->value());
    int value;
    if (!(iss >> value) || !iss.eof() || value < 0) {
        EvoRegression::Log::get_logger()->error("Invalid input. Please enter an integer number.");
    }
    else {
        T->migration_ratio = value;
        EvoRegression::Log::get_logger()->info("Migration ratio set to: " + std::to_string(T->migration_ratio));
    }
}

/**
 * Callback function for handling the migration interval input.
 * Parses the input value as an integer and sets it as the migration interval.
 * If the input is invalid (not an integer or negative), an error message is logged.
 * @param w The Fl_Widget triggering the callback.
 * @param v The user data associated with the widget.
 */
void EvoView::migration_interval_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    std::istringstream iss(((Fl_Input*)w)->value());
    int value;
    if (!(iss >> value) || !iss.eof() || value < 0) {
        EvoRegression::Log::get_logger()->error("Invalid input. Please enter an integer number.");
    }
    else {
        T->migration_interval = value;
        EvoRegression::Log::get_logger()->info("Migration interval set to: " + std::to_string(T->migration_interval));
    }
}

/**
 * Callback function for the decomposition method choice box.
 * Updates the decomposition_method field in the EvoView object and logs the new value.
 *
 * @param w The Fl_Widget triggering the callback.
 * @param v A pointer to the EvoView object.
 */
void EvoView::decomposition_choice_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    T->decomposition_method = ((Fl_Choice*)w)->text();
    T->evo_api.set_solver(T->decomposition_method);
}

/**
 * Callback function for the load file button in EvoView.
 * This function is triggered when the load file button is clicked.
 * It retrieves the file path from the EvoView instance, sets the boundary conditions using the Evo API,
 * and loads the file using the Evo API.
 *
 * @param w The Fl_Widget object representing the load file button.
 * @param v A pointer to the EvoView instance.
 */
void EvoView::load_file_button_callback(Fl_Widget* /*w*/, void* v) {
    EvoView* T = (EvoView*)v;
    T->get_filepath();
    T->evo_api.set_boundary_conditions(T->generations_size, T->generations_count, T->interference_size, T->mutation_rate, T->island_count, T->migration_ratio, T->migration_interval);
    T->evo_api.load_file(T->filepath);
}

/**
 * Callback function for the batch predict button in the EvoView class.
 * It checks if the Evo API is ready to predict and starts a new thread to perform batch prediction.
 * If the Evo API is not ready, it logs an error message.
 *
 * @param w The Fl_Widget object that triggered the callback (not used).
 * @param v A pointer to the EvoView object.
 */
void EvoView::batch_predict_button_callback(Fl_Widget* /*w*/, void* v) {
    EvoView* T = (EvoView*)v;

    if (!T->evo_api.is_ready_to_predict()) {
        EvoRegression::Log::get_logger()->error("Cannot start prediction. Please load a file first.");
        return;
    }

    std::thread predict_thread(
        &EvoView::call_batch_predict,
        T,
        T->evo_api
    );

    predict_thread.detach();
}

//*************************************************************************************************methods

/**
 * Calls the batch_predict method of the EvoAPI object.
 *
 * @param evo_api The EvoAPI object to call the batch_predict method on.
 */
void EvoView::call_batch_predict(EvoAPI evo_api) {
    try {
        evo_api.batch_predict();
    }
    catch (const std::exception& e) {
        EvoRegression::Log::get_logger()->error("Standard exception caught in batch predict method call: " + std::string(e.what()));
    }
    catch (...) {
        EvoRegression::Log::get_logger()->error("Unknown exception caught in batch predict method call.");
    }
};

/**
 * @brief Gets the file path from the user.
 *
 * This function displays a file chooser dialog and gets the file path from the user.
 * It sets the `filepath` field to the selected file path and logs the result.
 */
void EvoView::get_filepath() {
    Fl_File_Chooser chooser(".", "*", Fl_File_Chooser::SINGLE, "Select a file");
    chooser.show();
    while (chooser.shown()) {
        Fl::wait();
    }
    if (chooser.value() != nullptr) {
        filepath = chooser.value();
        EvoRegression::Log::get_logger()->info("Filepath set to: " + filepath);
    }
}

/**
 * Connects the terminal sink to the global logger.
 * This function creates a FLTK sink terminal and tries to connect to global logger.
 */
void EvoView::connect_terminal_to_logger() {
    // FLTK sink terminal
    auto fltk_terminal_sink = std::make_shared<Fl_Terminal_Sink<std::mutex>>(log_terminal);
    
    // Get global logger
    auto global_logger = spdlog::default_logger();

    if (!global_logger) {
        EvoRegression::Log::get_logger()->info("Cannot connect GUI terminal to global logger.");
    }
    else {
        global_logger->sinks().push_back(fltk_terminal_sink);
        EvoRegression::Log::get_logger()->info("EvoView logger connected to global logger.");
    }
}

/**
 * @brief Creates a new menu bar.
 *
 * This function creates a new menu bar and adds the "File/Quit" and "Help" items to it.
 * It also sets the keyboard shortcuts for the menu items.
 *
 * @param width The width of the menu bar in pixels.
 * @param height The height of the menu bar in pixels.
 * @return Fl_Menu_Bar* A pointer to the newly created menu bar.
 */
Fl_Menu_Bar* EvoView::create_menu_bar(int width, int height) {
    Fl_Menu_Bar* bar = new Fl_Menu_Bar(0, 0, width, height);
    bar->add("File/Quit", FL_CTRL + 'q', quit_callback);
    bar->add("Help", 0, help_callback);
    return bar;
}

/**
 * @brief Creates a new terminal.
 *
 * This function creates a new terminal.
 *
 * @param x The x-coordinate of the terminal.
 * @param y The y-coordinate of the terminal.
 * @param w The width of the terminal in pixels.
 * @param h The height of the terminal in pixels.
 * @return Fl_Terminal* A pointer to the newly created terminal.
 */
Fl_Terminal* EvoView::create_terminal(int x, int y, int w, int h) {
    Fl_Terminal* terminal = new Fl_Terminal(x, y, w, h);
    terminal->history_lines(1000);
    return terminal;
}

/**
 * @brief Creates a new pack widget.
 *
 * This function creates a new pack widget.
 *
 * @param x The x-coordinate of the pack widget.
 * @param y The y-coordinate of the pack widget.
 * @param w The width of the pack widget in pixels.
 * @param h The height of the pack widget in pixels.
 * @return Fl_Pack* A pointer to the newly created pack widget.
 */
Fl_Pack* EvoView::create_main_widget_pack(int x, int y, int w, int h) {
    return new Fl_Pack(x, y, w, h);
}

/**
 * @brief Creates a new Fl_Box with the specified label.
 *
 * This function creates a new Fl_Box with a specified height and label. The box is aligned to the left and inside.
 *
 * @param height The height of the box.
 * @param label The label to display in the box.
 * @return Fl_Box* A pointer to the newly created Fl_Box.
 */
Fl_Box* EvoView::create_label(int height, const char* label) {
    Fl_Box* box = new Fl_Box(0, 0, 0, height, label);
    box->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
    box->box(FL_UP_BOX);
    return box;
}

/**
 * @brief Creates a new button.
 *
 * This function creates a new button.
 *
 * @param h The height of the button in pixels.
 * @return Fl_Button* A pointer to the newly created button.
 */
Fl_Button* EvoView::create_load_button(int h) {
    Fl_Button* button = new Fl_Button(0, 0, 0, h, "Create data");
    button->callback(load_file_button_callback, (void*)this);
    button->tooltip("Loads data from a file and adds interference columns according to boundary conditions to dataset. The file must be in the CSV format.");
    return button;
}

/**
 * @brief Creates a Fl_Button object with the specified dimensions and label.
 *
 * @param x The x-coordinate of the button.
 * @param y The y-coordinate of the button.
 * @param w The width of the button.
 * @param h The height of the button.
 * @param label The label text of the button.
 * @return Fl_Button* A pointer to the created Fl_Button object.
 */
Fl_Button* EvoView::create_batch_predict_button(int h) {
    Fl_Button* button = new Fl_Button(0, 0, 0, h, "Start batch prediction");
    button->callback(batch_predict_button_callback, (void*)this);
    button->tooltip("Starts the batch prediction process.");
    return button;
}

/**
 * @brief Creates a new combo box.
 *
 * This function creates a new combo box.
 *
 * @param h The height of the combo box in pixels.
 * @return Fl_Choice* A pointer to the newly created combo box.
 */
Fl_Choice* EvoView::create_combo_box(int h) {
    Fl_Choice* combo_box = new Fl_Choice(0, 0, 0, h);
    combo_box->add("LLT");
    combo_box->add("LDLT");
    combo_box->add("ColPivHouseholderQr");
    combo_box->value(1);
    combo_box->tooltip("Choose the decomposition method");
    combo_box->callback(decomposition_choice_callback, (void*)this);
    return combo_box;
}

/**
 * @brief Creates a new input box.
 *
 * This function creates a new input box.
 *
 * @param h The height of the input box in pixels.
 * @return Fl_Input* A pointer to the newly created input box.
 */
Fl_Input* EvoView::create_gen_count_box(int h) {
    Fl_Input* inputBox = new Fl_Input(0, 0, 0, h);
    inputBox->value(generations_count);
    inputBox->tooltip("Enter the generations count here");
    inputBox->callback(gen_count_input_callback, (void*)(this));
    return inputBox;
}

/**
 * @brief Creates a new input box.
 *
 * This function creates a new input box.
 *
 * @param h The height of the input box in pixels.
 * @return Fl_Input* A pointer to the newly created input box.
 */
Fl_Input* EvoView::create_gen_size_box(int h) {
    Fl_Input* inputBox = new Fl_Input(0, 0, 0, h);
    inputBox->value(generations_size);
    inputBox->tooltip("Enter the size of the generation here");
    inputBox->callback(gen_size_input_callback, (void*)(this));
    return inputBox;
}

/**
 * @brief Creates a new input box.
 *
 * This function creates a new input box.
 *
 * @param h The height of the input box in pixels.
 * @return Fl_Input* A pointer to the newly created input box.
 */
Fl_Input* EvoView::create_inter_size_box(int h) {
    Fl_Input* inputBox = new Fl_Input(0, 0, 0, h);
    inputBox->value(interference_size);
    inputBox->tooltip("Enter the number of interference columns here");
    inputBox->callback(gen_interference_size_callback, (void*)(this));
    return inputBox;
}

/**
 * @brief Creates a new input box for entering the mutation rate.
 *
 * This function creates a new Fl_Input object, sets its initial value to the current mutation rate,
 * sets its tooltip to "Enter the mutation rate here [0 - 1]", and sets its callback to mutation_rate_callback.
 *
 * @param h The height of the input box.
 * @return Fl_Input* A pointer to the new input box.
 */
Fl_Input* EvoView::create_mutation_rate_box(int h) {
    Fl_Input* inputBox = new Fl_Input(0, 0, 0, h);
    inputBox->value(mutation_rate);
    inputBox->tooltip("Enter the mutation ratio here 0 - 100 [%]");
    inputBox->callback(mutation_rate_callback, (void*)(this));
    return inputBox;
}

/**
 * @brief Creates a Fl_Input object with the specified dimensions and sets its initial value.
 *
 * @param x The x-coordinate of the input box.
 * @param y The y-coordinate of the input box.
 * @param w The width of the input box.
 * @param h The height of the input box.
 * @return Fl_Input* A pointer to the created Fl_Input object.
 */
Fl_Input* EvoView::create_island_count_box(int h) {
    Fl_Input* inputBox = new Fl_Input(0, 0, 0, h);
    inputBox->value(island_count);
    inputBox->tooltip("Enter the island count here [integer]");
    inputBox->callback(island_count_callback, (void*)(this));
    return inputBox;
}

/**
 * @brief Creates a Fl_Input object with the specified dimensions and sets its initial value.
 *
 * @param x The x-coordinate of the input box.
 * @param y The y-coordinate of the input box.
 * @param w The width of the input box.
 * @param h The height of the input box.
 * @return Fl_Input* A pointer to the created Fl_Input object.
 */
Fl_Input* EvoView::create_migration_ratio_box(int h) {
    Fl_Input* inputBox = new Fl_Input(0, 0, 0, h);
    inputBox->value(migration_ratio);
    inputBox->tooltip("Enter the migration_ratio here [integer]");
    inputBox->callback(migration_ratio_callback, (void*)(this));
    return inputBox;
}

/**
 * @brief Creates a Fl_Input widget with the specified height.
 *
 * @param h The height of the Fl_Input widget.
 * @return Fl_Input* A pointer to the created Fl_Input widget.
 */
Fl_Input* EvoView::create_migration_interval_box(int h) {
    Fl_Input* inputBox = new Fl_Input(0, 0, 0, h);
    inputBox->value(migration_interval);
    inputBox->tooltip("Enter the migration_interval here [integer]");
    inputBox->callback(migration_interval_callback, (void*)(this));
    return inputBox;
}

/**
 * @brief Creates an invisible Fl_Box to be used as a spacer.
 *
 * This function creates a new Fl_Box with a specified height and makes it invisible. This can be used to create space between widgets in a layout.
 *
 * @param height The height of the spacer.
 * @return Fl_Box* A pointer to the newly created invisible Fl_Box.
 */
Fl_Box* EvoView::create_spacer(int height) {
    Fl_Box* spacer = new Fl_Box(0, 0, 0, height);
    spacer->box(FL_NO_BOX); // Make the box invisible
    return spacer;
}

/**
 * @brief Renders the main window.
 *
 * This function renders the main window. It creates the menu bar, the terminal, and the main widget pack.
 * It also creates the buttons, input boxes, and check buttons and adds them to the main widget pack.
 */
void EvoView::render_main_window() {
    this->begin();
    {
        menu_bar = create_menu_bar(WINDOW_WIDTH, MENU_HEIGHT);
        log_terminal = create_terminal(MAIN_WIDGET_PACK_WIDTH, MENU_HEIGHT,
            WINDOW_WIDTH - MAIN_WIDGET_PACK_WIDTH,
            WINDOW_HEIGHT - MENU_HEIGHT
        );
        main_widget_pack = create_main_widget_pack(
            0,
            MENU_HEIGHT,
            MAIN_WIDGET_PACK_WIDTH,
            WINDOW_HEIGHT - MENU_HEIGHT
        );
        main_widget_pack->begin();
        {
            gen_count_label = create_label(LABEL_HEIGHT, "Generations count:");
            gen_count_box = create_gen_count_box(BUTTON_HEIGHT);
            gen_size_label = create_label(LABEL_HEIGHT, "Generation size:");
            gen_size_box = create_gen_size_box(BUTTON_HEIGHT);
            inter_size_label = create_label(LABEL_HEIGHT, "Interference columns:");
            inter_size_box = create_inter_size_box(BUTTON_HEIGHT);
            mutation_rate_label = create_label(LABEL_HEIGHT, "Mutation rate:");
            mutation_rate_box = create_mutation_rate_box(BUTTON_HEIGHT);
            island_count_label = create_label(LABEL_HEIGHT, "Island count:");
            island_count_box = create_island_count_box(BUTTON_HEIGHT);
            migration_ratio_label = create_label(LABEL_HEIGHT, "Migration ratio:");
            migration_ratio_box = create_migration_ratio_box(BUTTON_HEIGHT);
            migration_interval_label = create_label(LABEL_HEIGHT, "Migration interval:");
            migration_interval_box = create_migration_interval_box(BUTTON_HEIGHT);
            load_button = create_load_button(BUTTON_HEIGHT);
            decomposition_choice_chbox = create_combo_box(BUTTON_HEIGHT);
            batch_predict_button = create_batch_predict_button(BUTTON_HEIGHT);
        }
        main_widget_pack->end();
    }
    this->end();
}

/**
 * @brief Sets the appearance of the application.
 *
 * This function sets the appearance of the application. It sets the GTK+ theme, disables the focus
 * rectangle, enables drag-and-drop text, and sets the font.
 */
void EvoView::set_appearance() {
    Fl::scheme("gtk+");
    Fl::option(Fl::OPTION_VISIBLE_FOCUS, false);
    Fl::option(Fl::OPTION_DND_TEXT, true);
    Fl::set_fonts("Roboto-Regular.ttf");
    Fl::set_boxtype(FL_UP_BOX, FL_ENGRAVED_BOX);
    this->resizable(this);
}

//*************************************************************************************************Fl_Terminal_Sink

/**
 * @brief Constructs a new Fl_Terminal_Sink object.
 *
 * This constructor initializes a new Fl_Terminal_Sink object, which represents a sink for the logger.
 * It sets the `log_terminal` field to the provided terminal.
 *
 * @param terminal A pointer to the terminal.
 */
template <typename T>
Fl_Terminal_Sink<T>::Fl_Terminal_Sink(Fl_Terminal* terminal) : log_terminal(terminal) {};

template <typename T>
void Fl_Terminal_Sink<T>::sink_it_(const spdlog::details::log_msg& msg) {
    spdlog::memory_buf_t formatted;
    spdlog::sinks::base_sink<T>::formatter_->format(msg, formatted);
    std::string formatted_str = fmt::to_string(formatted);
    log_terminal->append(formatted_str.c_str());
};

template <typename T>
void Fl_Terminal_Sink<T>::flush_() {
    // FLTK doesn't require flushing, but you can implement this if needed
};

//*************************************************************************************************explicit instantiations
template class Fl_Terminal_Sink<std::mutex>;









