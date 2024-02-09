#pragma once
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
#include "IEvoAPI.hpp"

class EvoView : public Fl_Window {

    // api
    std::unique_ptr<IEvoAPI> _api;

    //flags
    std::string decomposition_method;

    //variables
    std::string filepath;
    
    int generations_count;
    int generations_size;
    int interference_size;
    int mutation_rate;
    int basis_function_complexity;
    float regularization_parameter;
    int island_count;
    int migration_ratio;
    int migration_interval;
    int test_ratio;

    //widgets

    Fl_Box* spacer;
    Fl_Menu_Bar* menu_bar;
    Fl_Terminal* log_terminal;
    Fl_Pack* main_widget_pack;
    Fl_Button* load_button;
    Fl_Button* batch_predict_button;
    Fl_Box* basis_function_complexity_label;
    Fl_Input* basis_function_complexity_box;
    Fl_Box* regularization_parameter_label;
    Fl_Input* regularization_parameter_box;
    Fl_Box* mutation_rate_label;
    Fl_Input* mutation_rate_box;
    Fl_Box* test_ratio_label;
    Fl_Input* test_ratio_box;
    Fl_Box* gen_count_label;
    Fl_Input* gen_count_box;
    Fl_Box* gen_size_label;
    Fl_Input* gen_size_box;
    Fl_Box* inter_size_label;
    Fl_Input* inter_size_box;
    Fl_Box* island_count_label;
    Fl_Input* island_count_box;
    Fl_Box* migration_ratio_label;
    Fl_Input* migration_ratio_box;
    Fl_Box* migration_interval_label;
    Fl_Input* migration_interval_box;
    Fl_Choice* decomposition_choice_chbox;
    Fl_Check_Button* export_log_checkbutton;

    //callbacks

    static void quit_callback(Fl_Widget* /*w*/, void* /*data*/);
    static void help_callback(Fl_Widget* /*w*/, void* /*data*/);
    static void gen_count_input_callback(Fl_Widget* w, void* v);
    static void gen_size_input_callback(Fl_Widget* w, void* v);
    static void gen_interference_size_callback(Fl_Widget* w, void* v);
    static void mutation_rate_callback(Fl_Widget* w, void* v);
    static void basis_function_complexity_callback(Fl_Widget* w, void* v);
    static void regularization_parameter_callback(Fl_Widget* w, void* v);
    static void test_ratio_callback(Fl_Widget* w, void* v);
    static void island_count_callback(Fl_Widget* w, void* v);
    static void migration_ratio_callback(Fl_Widget* w, void* v);
    static void migration_interval_callback(Fl_Widget* w, void* v);
    static void decomposition_choice_callback(Fl_Widget* w, void* v);
    static void load_file_button_callback(Fl_Widget* w, void* v);
    static void batch_predict_button_callback(Fl_Widget* /*w*/, void* v);

    //methods

    Fl_Box* create_spacer(int height);
    Fl_Menu_Bar* create_menu_bar(int width, int height);
    Fl_Terminal* create_terminal(int x, int y, int w, int h);
    Fl_Pack* create_main_widget_pack(int x, int y, int w, int h);
    Fl_Button* create_load_button(int h);
    Fl_Button* create_batch_predict_button(int h);
    Fl_Input* create_basis_function_complexity_box(int h);
    Fl_Input* create_regularization_parameter_box(int h);
    Fl_Input* create_test_ratio_box(int h);
    Fl_Choice* create_combo_box(int h);
    Fl_Input* create_gen_count_box(int h);
    Fl_Input* create_gen_size_box(int h);
    Fl_Input* create_inter_size_box(int h);
    Fl_Input* create_mutation_rate_box(int h);
    Fl_Input* create_island_count_box(int h);
    Fl_Input* create_migration_ratio_box(int h);
    Fl_Input* create_migration_interval_box(int h);
    Fl_Box* create_label(int height, const char* label);

    void get_filepath();
    void set_appearance();
    void connect_terminal_to_logger();
    void render_main_window();
    void call_batch_predict();

public:
    
    EvoView(const char* title);
    void bind_to_backend(std::unique_ptr<IEvoAPI>);
};



/**
 * @brief A custom sink for logging messages to a Fl_Terminal widget.
 * 
 * This class is a template class that inherits from spdlog::sinks::base_sink<T>.
 * It provides functionality to sink log messages to a Fl_Terminal widget.
 * 
 * @tparam T The type of log messages to be sunk.
 */
template <typename T>
class Fl_Terminal_Sink : public spdlog::sinks::base_sink<T> {

    Fl_Terminal* log_terminal;

protected:

    void sink_it_(const spdlog::details::log_msg& msg) override;
    void flush_() override;

public:

    Fl_Terminal_Sink(Fl_Terminal* terminal);
};