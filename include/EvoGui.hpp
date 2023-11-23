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
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

class EvoView : public Fl_Window {

    // logger

    std::shared_ptr<spdlog::logger> logger;

    //flags

    std::string decomposition_method;
    bool export_log_file;

    //variables

    int generations_count;
    int generations_size;
    int interference_size;

    //widgets

    Fl_Menu_Bar* menu_bar;
    Fl_Terminal* log_terminal;
    Fl_Pack* main_widget_pack;
    Fl_Button* load_button;
    Fl_Button* predict_button;
    Fl_Input* gen_count_box;
    Fl_Input* gen_size_box;
    Fl_Input* inter_size_box;
    Fl_Choice* decomposition_choice_chbox;
    Fl_Check_Button* export_log_checkbutton;

    //callbacks

    static void quit_callback(Fl_Widget* /*w*/, void* /*data*/);
    static void help_callback(Fl_Widget* /*w*/, void* /*data*/);
    static void gen_count_input_callback(Fl_Widget* w, void* v);
    static void gen_size_input_callback(Fl_Widget* w, void* v);
    static void gen_interference_size_callback(Fl_Widget* w, void* v);
    static void export_file_callback(Fl_Widget* w, void* v);
    static void decomposition_choice_callback(Fl_Widget* w, void* v);

    //methods

    Fl_Menu_Bar* create_menu_bar(int width, int height);
    Fl_Terminal* create_terminal(int x, int y, int w, int h);
    Fl_Pack* create_main_widget_pack(int x, int y, int w, int h);
    Fl_Button* create_load_button(int h);
    Fl_Button* create_predict_button(int h);
    Fl_Choice* create_combo_box(int h);
    Fl_Input* create_gen_count_box(int h);
    Fl_Input* create_gen_size_box(int h);
    Fl_Input* create_inter_size_box(int h);
    Fl_Check_Button* create_export_file_box(int h);

    void set_appearance();
    void init_logger();

public:
    EvoView(int width, int height, const char* title);
    void render_main_window();
};

template <typename T>
class Fl_Terminal_Sink : public spdlog::sinks::base_sink<T> {

    Fl_Terminal* log_terminal;

protected:

    void sink_it_(const spdlog::details::log_msg& msg) override;

    void flush_() override;

public:

    Fl_Terminal_Sink(Fl_Terminal* terminal);
};