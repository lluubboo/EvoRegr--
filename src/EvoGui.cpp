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
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <string>
#include "EvoGui.hpp"

//*************************************************************************************************constants

static constexpr const char* WINDOW_NAME = "EVOREGR++ 1.0.0";
static constexpr int WINDOW_WIDTH = 1500;
static constexpr int WINDOW_HEIGHT = 600;
static constexpr int MAIN_WIDGET_PACK_WIDTH = 200;
static constexpr int BUTTON_HEIGHT = 30;
static constexpr int SPACER_HEIGHT = 0;
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
EvoView::EvoView(int width, int height, const char* title) : Fl_Window(width, height, title), decomposition_method{ "LDLT" }, export_log_file{ false }, generations_count { 100 }, generations_size{ 100 }, interference_size{ 0 } {
    set_appearance();
    render_main_window();
    init_logger();
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
    window->resizable(*disp);
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
        T->logger->info("Generations count set to: " + std::to_string(T->generations_count));
    }
    else {
        T->logger->error("Invalid input. Please enter a valid integer.");
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
        T->logger->info("Generations size set to: " + std::to_string(T->generations_size));
    }
    else {
        T->logger->error("Invalid input. Please enter a valid integer.");
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
        T->logger->info("Interference columns size set to: " + std::to_string(T->interference_size));
    }
    else {
        T->logger->error("Invalid input. Please enter a valid integer.");
    }
}

void EvoView::export_file_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    T->export_log_file = ((Fl_Check_Button*)w)->value();
    T->logger->info("Export log file set to: " + std::to_string(T->export_log_file));
}

/**
 * @brief Handles the state change of the export file checkbox.
 *
 * This function is called when the user changes the state of the export file checkbox.
 * It updates the `export_log_file` field with the new state of the checkbox and logs a message.
 *
 * @param w The FLTK widget (an Fl_Check_Button instance) that triggered the callback.
 * @param v A pointer to the EvoView instance.
 */
void EvoView::decomposition_choice_callback(Fl_Widget* w, void* v) {
    EvoView* T = (EvoView*)v;
    T->decomposition_method = ((Fl_Choice*)w)->text();
    T->logger->info("Decomposition method set to: " + T->decomposition_method);
}

//*************************************************************************************************methods

/**
 * @brief Initializes the logger.
 *
 * This function initializes the logger. It creates a new Fl_Terminal_Sink instance and passes it to
 * the logger. It also sets the logger pattern and level.
 */
void EvoView::init_logger() {
    // FLTK sink
    auto fltk_sink = std::make_shared<Fl_Terminal_Sink<std::mutex>>(log_terminal);
    // logger
    logger = std::make_shared<spdlog::logger>("EvoView", spdlog::sinks_init_list{ fltk_sink });
    // register the logger so it can be accessed using spdlog::get()
    spdlog::register_logger(logger);
    // settings
    spdlog::set_pattern("[EvoView] [%H:%M:%S] [%^%l%$] [thread %t] %v");
    spdlog::set_level(spdlog::level::debug);
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
    return new Fl_Terminal(x, y, w, h);
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
 * @brief Creates a new button.
 *
 * This function creates a new button.
 *
 * @param h The height of the button in pixels.
 * @return Fl_Button* A pointer to the newly created button.
 */
Fl_Button* EvoView::create_load_button(int h) {
    return new Fl_Button(0, 0, 0, h, "Load data");
}

/**
 * @brief Creates a new button.
 *
 * This function creates a new button.
 *
 * @param h The height of the button in pixels.
 * @return Fl_Button* A pointer to the newly created button.
 */
Fl_Button* EvoView::create_predict_button(int h) {
    return new Fl_Button(0, 0, 0, h, "Start prediction");
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
    combo_box->add("ColPivHouseholderQR");
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
 * @brief Creates a new check button.
 *
 * This function creates a new check button.
 *
 * @param h The height of the check button in pixels.
 * @return Fl_Check_Button* A pointer to the newly created check button.
 */
Fl_Check_Button* EvoView::create_export_file_box(int h) {
    Fl_Check_Button* checkbox = new Fl_Check_Button(0, 0, 0, h, "export log file");
    checkbox->align(FL_ALIGN_INSIDE);
    checkbox->tooltip("Check this box to enable log file export before running the prediction.");
    checkbox->callback(export_file_callback, (void*)this);
    return checkbox;
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
            load_button = create_load_button(BUTTON_HEIGHT);
            gen_count_box = create_gen_count_box(BUTTON_HEIGHT);
            gen_size_box = create_gen_size_box(BUTTON_HEIGHT);
            inter_size_box = create_inter_size_box(BUTTON_HEIGHT);
            export_log_checkbutton = create_export_file_box(BUTTON_HEIGHT);
            decomposition_choice_chbox = create_combo_box(BUTTON_HEIGHT);
            predict_button = create_predict_button(BUTTON_HEIGHT);
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









